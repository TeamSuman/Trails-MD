"""MSM-guided / least-counts spawner.

Selects restart frames that most reduce the statistical error of the Markov State
Model. Two modes:

* **MSM-guided** (when the orchestrator supplies the latest ``MSMResult`` and the
  estimator's clustering): score each microstate by
  ``π_i · |ψ_i| · (σ_out,i / mean) + α/√c_i`` — its stationary flux × leverage on
  the slow processes (slow-eigenvector amplitude) × outflow statistical
  uncertainty (Dirichlet σ of the transition row), plus a least-counts/frontier
  exploration term. New / disconnected microstates get the exploration weight so
  they get connected. This throws runs at the transitions whose in/out rates are
  both uncertain and important.
* **Least-counts fallback** (no MSM yet — iteration 0 / after resume): the classic
  inverse-count weighting on an independent clustering.

Implements the standard ``sample(points, top_n, history)`` contract, returning
indices into the cumulative point cloud, so it is a drop-in ``spawn_scheme: msm``.
With ``alpha`` large or ``uncertainty=False`` and uniform leverage the MSM-guided
score reduces to least-counts (backward-compatible default behaviour).
"""

from __future__ import annotations

import numpy as np

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class MSMSpawner(Spawner):
    """Microstate spawner targeting MSM statistical convergence.

    Parameters
    ----------
    n_clusters:
        Microstate count for the least-counts fallback clustering.
    mode / target:
        ``"explore"`` (default) or ``"target"`` (bias toward ``target``).
    weighting:
        Fallback least-counts weighting (``"least_counts"`` or ``"sqrt"``).
    alpha:
        Weight of the exploration / least-counts term in the MSM-guided score.
    leverage:
        Number of slow eigenvectors used for the leverage factor (0 → uniform).
    uncertainty:
        Include the outflow-uncertainty factor (``True``) or not.
    seed:
        RNG / clustering seed.
    """

    def __init__(
        self,
        n_clusters: int = 150,
        mode: str = "explore",
        target: list | None = None,
        weighting: str = "least_counts",
        alpha: float = 1.0,
        leverage: int = 1,
        uncertainty: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.n_clusters = int(n_clusters)
        self.mode = mode
        self.target = np.asarray(target, dtype=float) if target is not None else None
        self.weighting = weighting
        self.alpha = float(alpha)
        self.leverage = int(leverage)
        self.uncertainty = bool(uncertainty)
        # Set by the orchestrator each iteration (previous iteration's MSM and the
        # estimator's clustering); None → least-counts fallback.
        self.msm_result = None
        self.cluster_model = None

    # ------------------------------------------------------------------ #
    def sample(self, points: np.ndarray, top_n: int, history=None) -> list:
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            raise ValueError("Cannot sample MSM points from an empty point cloud.")
        cumulative = _cumulative_points(points, history)
        if cumulative.ndim == 1:
            cumulative = cumulative.reshape(-1, 1)
        if len(cumulative) == 1:
            return [0 for _ in range(top_n)]

        frame_weights = self._msm_guided_weights(cumulative)
        if frame_weights is None:
            frame_weights = self._least_counts_weights(cumulative)

        if self.mode == "target" and self.target is not None:
            dists = np.linalg.norm(cumulative - self.target, axis=1)
            frame_weights = frame_weights / (dists + 1e-10)

        total = frame_weights.sum()
        if total <= 0 or not np.isfinite(total):
            frame_weights = np.ones(len(cumulative)) / len(cumulative)
        else:
            frame_weights = frame_weights / total

        n_cumulative = len(cumulative)
        n_nonzero = int(np.count_nonzero(frame_weights))
        replace = n_cumulative < top_n or n_nonzero < top_n
        return (
            self.rng.choice(
                np.arange(n_cumulative), size=top_n, replace=replace, p=frame_weights
            )
            .astype(int)
            .tolist()
        )

    # ------------------------------------------------------------------ #
    def _msm_guided_weights(self, cumulative: np.ndarray) -> np.ndarray | None:
        """Per-frame weights from uncertainty × leverage × flux, or None to fall back."""
        res = self.msm_result
        model = self.cluster_model
        if res is None or model is None:
            return None
        T = getattr(res, "transition_matrix", None)
        pi = getattr(res, "stationary_distribution", None)
        counts = getattr(res, "counts_per_state", None)
        symbols = getattr(res, "state_symbols", None)
        if T is None or pi is None or counts is None or symbols is None:
            return None
        try:
            micro = np.asarray(model.transform(cumulative), dtype=int)
        except Exception:  # noqa: BLE001 - clustering mismatch → fall back
            return None

        T = np.asarray(T, dtype=float)
        pi = np.asarray(pi, dtype=float)
        counts = np.asarray(counts, dtype=float)
        symbols = np.asarray(symbols, dtype=int)
        n_active = len(pi)
        if n_active == 0 or len(counts) != n_active:
            return None

        # Outflow uncertainty per active state (Dirichlet row variance).
        var = T * (1.0 - T) / (counts[:, None] + 1.0)
        s_out = np.sqrt(np.clip(var.sum(axis=1), 0.0, None))
        mean_s = s_out[s_out > 0].mean() if np.any(s_out > 0) else 1.0
        s_term = (s_out / mean_s) if self.uncertainty else np.ones(n_active)

        # Leverage = summed |slow right-eigenvector| amplitude.
        E = getattr(res, "eigenvectors", None)
        if E is not None and self.leverage > 0:
            E = np.asarray(E, dtype=float)
            lev = np.abs(E[:, : self.leverage]).sum(axis=1)
        else:
            lev = np.ones(n_active)

        score_active = pi * lev * s_term
        micro_base = score_active + self.alpha / np.sqrt(counts + 1.0)  # per active state

        symbol_to_active = {int(s): i for i, s in enumerate(symbols)}
        active_idx = np.array(
            [symbol_to_active.get(int(m), -1) for m in micro], dtype=int
        )
        is_active = active_idx >= 0

        # Microstate-level base weight: active states use their score; frontier
        # (new / disconnected) microstates get a flat exploration weight.
        base = np.full(len(cumulative), self.alpha, dtype=float)
        if np.any(is_active):
            base[is_active] = micro_base[active_idx[is_active]]

        # Distribute the microstate weight over its frames in the cumulative cloud.
        _, inverse, sizes = np.unique(micro, return_inverse=True, return_counts=True)
        size_per_frame = sizes[inverse].astype(float)
        return base / np.maximum(size_per_frame, 1.0)

    # ------------------------------------------------------------------ #
    def _least_counts_weights(self, cumulative: np.ndarray) -> np.ndarray:
        assignments, n_states = self._cluster(cumulative)
        counts = np.bincount(assignments, minlength=n_states).astype(float)
        microstate_counts = np.maximum(counts[assignments], 1.0)
        if self.weighting == "sqrt":
            microstate_weight = 1.0 / np.sqrt(microstate_counts)
        else:
            microstate_weight = 1.0 / microstate_counts
        return microstate_weight / microstate_counts

    def _cluster(self, cumulative: np.ndarray):
        n_states = min(self.n_clusters, len(cumulative))
        try:
            from deeptime.clustering import KMeans

            model = KMeans(
                n_clusters=n_states, max_iter=100, fixed_seed=self.seed, progress=None
            ).fit_fetch(cumulative)
            assignments = np.asarray(model.transform(cumulative), dtype=int)
            return assignments, n_states
        except Exception:  # noqa: BLE001 - fall back to scikit-learn / numpy
            pass
        try:
            from sklearn.cluster import KMeans as SKMeans

            model = SKMeans(n_clusters=n_states, n_init=10, random_state=self.seed)
            assignments = model.fit_predict(cumulative)
            return np.asarray(assignments, dtype=int), n_states
        except Exception:  # noqa: BLE001 - last-resort single bucket
            return np.zeros(len(cumulative), dtype=int), 1


SpawnerFactory.register("msm", MSMSpawner)
