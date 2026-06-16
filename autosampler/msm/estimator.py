"""Markov State Model estimation over the AutoSampler CV / latent space.

``MSMEstimator`` wraps :mod:`deeptime.markov` into a small, testable API that the
adaptive loop can call once per iteration:

    estimator = MSMEstimator(lagtime=10, n_microstates=100)
    result = estimator.fit(trajs)          # trajs: list of (n_frames_i, n_cv)

It performs: clustering of the (continuous, per-walker) projections into
microstates -> sliding-window transition counts -> restriction to the largest
connected set -> maximum-likelihood (or Bayesian) MSM -> implied timescales,
VAMP-2 score and PCCA+ metastable decomposition. The output is a serialisable
:class:`~autosampler.msm.diagnostics.MSMResult`.

``deeptime`` is imported lazily so that importing this module never hard-requires
it; a clear error is raised only when estimation is actually attempted.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .diagnostics import ITSResult, MSMResult

logger = logging.getLogger(__name__)

_VALID_CLUSTER = ("kmeans", "regspace")
_VALID_ESTIMATOR = ("mle", "bayesian")


def _require_deeptime():
    try:
        import deeptime  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MSM estimation requires the 'deeptime' package. Install it with "
            "`pip install deeptime` (it is already declared as an AutoSampler "
            "dependency)."
        ) from exc


def _as_traj_list(trajs: Sequence[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == 0:
            continue
        out.append(arr)
    if not out:
        raise ValueError("MSMEstimator received no non-empty trajectories.")
    return out


class MSMEstimator:
    """Estimate an MSM from a list of continuous CV trajectories.

    Parameters
    ----------
    lagtime:
        Lag time (in saved frames) used for the production MSM.
    n_microstates:
        Number of clusters (microstates) used to discretise the CV space.
    cluster_method:
        ``"kmeans"`` or ``"regspace"`` (regular-space clustering).
    estimator:
        ``"mle"`` (maximum likelihood) or ``"bayesian"`` (adds posterior error
        bars on the slow timescales via :class:`deeptime.markov.msm.BayesianMSM`).
    n_metastable:
        If set, run PCCA+ to coarse-grain into this many metastable states.
    n_timescales:
        Number of slow processes (implied timescales) to track.
    lagtimes:
        Optional lag-time ladder for an implied-timescale sweep; when provided,
        :meth:`fit` attaches an :class:`ITSResult` to the output.
    n_bayesian_samples:
        Posterior sample count for the Bayesian estimator.
    seed:
        Random seed forwarded to the clustering for reproducibility.
    """

    def __init__(
        self,
        lagtime: int = 10,
        n_microstates: int = 100,
        cluster_method: str = "kmeans",
        estimator: str = "mle",
        n_metastable: int | None = None,
        n_timescales: int = 3,
        lagtimes: Sequence[int] | None = None,
        n_bayesian_samples: int = 50,
        regspace_dmin: float | None = None,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self.lagtime = int(lagtime)
        self.n_microstates = int(n_microstates)
        self.cluster_method = str(cluster_method).lower()
        self.estimator = str(estimator).lower()
        self.n_metastable = None if n_metastable is None else int(n_metastable)
        self.n_timescales = int(n_timescales)
        self.lagtimes = [int(lt) for lt in lagtimes] if lagtimes else None
        self.n_bayesian_samples = int(n_bayesian_samples)
        self.regspace_dmin = regspace_dmin
        self.seed = int(seed)

        if self.lagtime <= 0:
            raise ValueError("MSM lagtime must be a positive integer.")
        if self.n_microstates <= 1:
            raise ValueError("n_microstates must be greater than 1.")
        if self.cluster_method not in _VALID_CLUSTER:
            raise ValueError(f"cluster_method must be one of {_VALID_CLUSTER}.")
        if self.estimator not in _VALID_ESTIMATOR:
            raise ValueError(f"estimator must be one of {_VALID_ESTIMATOR}.")

        self._cluster_model = None  # last fitted clustering (deeptime model)

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    def cluster(self, trajs: Sequence[np.ndarray]):
        """Discretise CV trajectories into microstate index trajectories.

        Returns ``(dtrajs, cluster_model)`` where ``dtrajs`` is a list of int
        arrays (one per input trajectory) and ``cluster_model`` exposes
        ``cluster_centers``.
        """
        _require_deeptime()
        traj_list = _as_traj_list(trajs)
        stacked = np.vstack(traj_list)
        n_clusters = min(self.n_microstates, stacked.shape[0])

        if self.cluster_method == "kmeans":
            from deeptime.clustering import KMeans

            estimator = KMeans(
                n_clusters=n_clusters,
                max_iter=200,
                fixed_seed=self.seed,
                progress=None,
            )
            self._cluster_model = estimator.fit_fetch(stacked)
        else:  # regspace
            from deeptime.clustering import RegularSpace

            dmin = self.regspace_dmin
            if dmin is None:
                # Heuristic: spread the requested microstate budget over the
                # data extent so regular-space clustering yields ~n_microstates.
                extent = float(np.linalg.norm(stacked.max(0) - stacked.min(0)))
                dmin = max(extent / max(n_clusters, 1), 1e-6)
            estimator = RegularSpace(dmin=dmin, max_centers=n_clusters)
            self._cluster_model = estimator.fit_fetch(stacked)

        dtrajs = [
            np.asarray(self._cluster_model.transform(t), dtype=np.int64)
            for t in traj_list
        ]
        return dtrajs, self._cluster_model

    # ------------------------------------------------------------------ #
    # MSM estimation
    # ------------------------------------------------------------------ #
    def _count_model(self, dtrajs: list[np.ndarray], lagtime: int):
        from deeptime.markov import TransitionCountEstimator

        count_mode = "effective" if self.estimator == "bayesian" else "sliding"
        counts = TransitionCountEstimator(
            lagtime=lagtime, count_mode=count_mode
        ).fit_fetch(dtrajs)
        return counts.submodel_largest()

    def _fit_msm(self, connected_counts):
        from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM

        if self.estimator == "bayesian":
            posterior = BayesianMSM(
                n_samples=self.n_bayesian_samples
            ).fit_fetch(connected_counts)
            return posterior
        return MaximumLikelihoodMSM().fit_fetch(connected_counts)

    def fit(self, trajs: Sequence[np.ndarray], iteration: int | None = None) -> MSMResult:
        """Cluster, estimate the MSM and return a serialisable result."""
        _require_deeptime()
        dtrajs, cluster_model = self.cluster(trajs)

        connected = self._count_model(dtrajs, self.lagtime)
        if connected.n_states < 2:
            raise RuntimeError(
                "MSM connected set has fewer than 2 states; sampling is too "
                "sparse or lagtime too large for a Markov model at this stage."
            )

        fitted = self._fit_msm(connected)
        msm, timescale_errors = self._unwrap(fitted)

        k = min(self.n_timescales, msm.n_states - 1)
        timescales = np.asarray(msm.timescales(k=k), dtype=float)

        vamp2 = self._safe_score(msm, dtrajs)
        n_meta, meta_assign, meta_pop = self._pcca(msm)

        its = None
        if self.lagtimes:
            its = self.implied_timescales(dtrajs, self.lagtimes)

        return MSMResult(
            lagtime=self.lagtime,
            n_microstates=int(getattr(cluster_model, "n_clusters", self.n_microstates)),
            n_states_active=int(msm.n_states),
            timescales=timescales,
            stationary_distribution=np.asarray(msm.stationary_distribution, dtype=float),
            transition_matrix=np.asarray(msm.transition_matrix, dtype=float),
            cluster_centers=np.asarray(cluster_model.cluster_centers, dtype=float),
            counts_per_state=np.asarray(connected.state_histogram, dtype=float),
            vamp2_score=vamp2,
            estimator=self.estimator,
            iteration=iteration,
            n_metastable=n_meta,
            metastable_assignments=meta_assign,
            metastable_populations=meta_pop,
            its=its,
            timescale_errors=timescale_errors,
        )

    def implied_timescales(
        self, dtrajs: list[np.ndarray], lagtimes: Sequence[int]
    ) -> ITSResult:
        """Estimate implied timescales across a ladder of lag times."""
        from deeptime.markov.msm import MaximumLikelihoodMSM
        from deeptime.util.validation import implied_timescales

        models = []
        used = []
        for lag in lagtimes:
            try:
                cm = self._count_model(dtrajs, int(lag))
                if cm.n_states < 2:
                    continue
                models.append(MaximumLikelihoodMSM().fit_fetch(cm))
                used.append(int(lag))
            except Exception as exc:  # noqa: BLE001 - skip unusable lag times
                logger.debug("ITS lagtime %s skipped: %s", lag, exc)
        if not models:
            return ITSResult(lagtimes=np.asarray([]), timescales=np.zeros((0, 0)))

        its = implied_timescales(models)
        n_proc = min(self.n_timescales, its.max_n_processes)
        matrix = np.full((len(its.lagtimes), n_proc), np.nan)
        for p in range(n_proc):
            matrix[:, p] = np.asarray(its.timescales_for_process(p), dtype=float)
        return ITSResult(lagtimes=np.asarray(its.lagtimes), timescales=matrix)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _unwrap(self, fitted):
        """Return ``(point_estimate_msm, timescale_errors_or_None)``."""
        if self.estimator != "bayesian":
            return fitted, None
        prior = fitted.prior
        try:
            k = min(self.n_timescales, prior.n_states - 1)
            sample_ts = np.array(
                [np.asarray(m.timescales(k=k), dtype=float) for m in fitted.samples]
            )
            errors = np.nanstd(sample_ts, axis=0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Bayesian timescale error estimation failed: %s", exc)
            errors = None
        return prior, errors

    @staticmethod
    def _safe_score(msm, dtrajs) -> float | None:
        try:
            return float(msm.score(dtrajs=dtrajs, r=2))
        except Exception as exc:  # noqa: BLE001 - scoring is best-effort
            logger.debug("VAMP-2 scoring failed: %s", exc)
            return None

    def _pcca(self, msm):
        if not self.n_metastable or self.n_metastable < 2:
            return None, None, None
        if msm.n_states <= self.n_metastable:
            return None, None, None
        try:
            pcca = msm.pcca(self.n_metastable)
            assignments = np.asarray(pcca.assignments, dtype=int)
            populations = np.asarray(
                [
                    msm.stationary_distribution[assignments == m].sum()
                    for m in range(self.n_metastable)
                ],
                dtype=float,
            )
            return self.n_metastable, assignments, populations
        except Exception as exc:  # noqa: BLE001 - PCCA+ is best-effort
            logger.debug("PCCA+ failed: %s", exc)
            return None, None, None


class MSMEstimatorFactory:
    """Registry for MSM estimator variants, mirroring SpawnerFactory/EngineFactory."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, estimator_cls: type) -> None:
        cls._registry[name] = estimator_cls

    @classmethod
    def get(cls, name: str = "default", **kwargs: Any) -> MSMEstimator:
        estimator_cls = cls._registry.get(name, MSMEstimator)
        return estimator_cls(**kwargs)


MSMEstimatorFactory.register("default", MSMEstimator)
