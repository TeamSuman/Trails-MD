"""MSM-uncertainty / least-counts spawner.

Selects restart frames that most reduce the statistical error of the Markov
State Model. It discretises the cumulative CV/latent point cloud into
microstates and biases restarts toward sparsely-sampled microstates
(least-counts adaptive sampling), the standard strategy for driving an MSM
toward convergence on its slow processes.

Implements the same ``sample(points, top_n, history)`` contract as the other
spawners, returning indices into the cumulative point cloud, so it is a drop-in
``spawn_scheme: msm`` option.
"""

from __future__ import annotations

import numpy as np

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class MSMSpawner(Spawner):
    """Least-counts microstate spawner targeting MSM statistical convergence.

    Parameters
    ----------
    n_clusters:
        Number of microstates used to discretise the explored space (reuses the
        ``voronoi_clusters`` config value via the shared spawner kwargs).
    mode:
        ``"explore"`` (default) or ``"target"`` to bias toward a target point.
    target:
        Target CV point used when ``mode == "target"``.
    weighting:
        ``"least_counts"`` weights frames by ``1/sqrt(count)`` of their
        microstate; ``"inverse_counts"`` uses ``1/count`` (more aggressive).
    seed:
        Seed for the clustering / sampling RNG.
    """

    def __init__(
        self,
        n_clusters: int = 150,
        mode: str = "explore",
        target: list | None = None,
        weighting: str = "least_counts",
        seed: int = 42,
        **_,
    ):
        self.n_clusters = int(n_clusters)
        self.mode = mode
        self.target = np.asarray(target, dtype=float) if target is not None else None
        self.weighting = weighting
        self.seed = int(seed)

    def sample(self, points: np.ndarray, top_n: int, history=None) -> list:
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            raise ValueError("Cannot sample MSM points from an empty point cloud.")
        cumulative = _cumulative_points(points, history)
        if cumulative.ndim == 1:
            cumulative = cumulative.reshape(-1, 1)
        n_cumulative = len(cumulative)
        if n_cumulative == 1:
            return [0 for _ in range(top_n)]

        assignments, n_states = self._cluster(cumulative)
        counts = np.bincount(assignments, minlength=n_states).astype(float)
        microstate_counts = np.maximum(counts[assignments], 1.0)

        # Operate at the MICROSTATE level: a microstate is chosen with
        # probability ``microstate_weight``; a frame within it is chosen
        # uniformly. Dividing the microstate weight by its size yields the
        # per-frame weight, so a sparse region that collapses into a single
        # microstate competes against whole microstates, not against every
        # frame of a densely-sampled basin.
        if self.weighting == "sqrt":
            microstate_weight = 1.0 / np.sqrt(microstate_counts)  # softened
        else:  # least_counts (default): select microstates inversely to counts
            microstate_weight = 1.0 / microstate_counts
        frame_weights = microstate_weight / microstate_counts

        if self.mode == "target" and self.target is not None:
            dists = np.linalg.norm(cumulative - self.target, axis=1)
            frame_weights = frame_weights / (dists + 1e-10)

        total = frame_weights.sum()
        if total <= 0 or not np.isfinite(total):
            frame_weights = np.ones(n_cumulative, dtype=float) / n_cumulative
        else:
            frame_weights = frame_weights / total

        rng = np.random.default_rng(self.seed)
        n_nonzero = int(np.count_nonzero(frame_weights))
        replace = n_cumulative < top_n or n_nonzero < top_n
        return (
            rng.choice(np.arange(n_cumulative), size=top_n, replace=replace, p=frame_weights)
            .astype(int)
            .tolist()
        )

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
