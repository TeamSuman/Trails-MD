from __future__ import annotations

import numpy as np

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class LOFSpawner(Spawner):
    """Samples frames with high Local Outlier Factor scores.

    When *mode* is ``"target"`` and a *target* point is provided, the LOF
    weights are multiplied by the normalised proximity to the target so
    that outlier frames near the target region are preferred.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        mode: str = "explore",
        target: list | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.target = np.asarray(target, dtype=float) if target is not None else None

    def sample(self, points: np.ndarray, top_n: int, history=None) -> list:
        points = np.asarray(points, dtype=float)
        n_points = len(points)
        if n_points == 0:
            raise ValueError("Cannot sample LOF points from an empty point cloud.")
        cumulative_points = _cumulative_points(points, history)
        n_cumulative = len(cumulative_points)
        if n_cumulative == 1:
            return [0 for _ in range(top_n)]

        n_neighbors = min(self.n_neighbors, n_cumulative - 1)
        try:
            from sklearn.neighbors import LocalOutlierFactor

            model = LocalOutlierFactor(n_neighbors=n_neighbors)
            model.fit(cumulative_points)
            lof_scores = model.negative_outlier_factor_
            weights = lof_scores.max() - lof_scores
        except ModuleNotFoundError:
            weights = _lof_weights(cumulative_points, n_neighbors)

        # Target-guided: multiply by proximity to target
        if self.mode == "target" and self.target is not None:
            dists_to_target = np.linalg.norm(cumulative_points - self.target, axis=1)
            # Avoid division by zero; invert so closer == higher weight
            proximity = 1.0 / (dists_to_target + 1e-10)
            prox_sum = proximity.sum()
            if prox_sum > 0:
                proximity = proximity / prox_sum
            weights = weights * proximity

        denom = weights.sum()
        if denom <= 0 or not np.isfinite(denom):
            weights = np.ones(n_cumulative, dtype=float) / n_cumulative
        else:
            weights = weights / denom
        # Need at least top_n non-zero entries for replace=False sampling
        n_nonzero = int(np.count_nonzero(weights))
        replace = n_cumulative < top_n or n_nonzero < top_n
        return (
            self.rng.choice(
                np.arange(n_cumulative), size=top_n, replace=replace, p=weights
            )
            .astype(int)
            .tolist()
        )


SpawnerFactory.register("lof", LOFSpawner)


def _lof_weights(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    """NumPy LOF approximation used when scikit-learn is unavailable."""
    distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    order = np.argsort(distances, axis=1)[:, 1 : n_neighbors + 1]
    sorted_distances = np.take_along_axis(
        distances, np.argsort(distances, axis=1), axis=1
    )
    k_distance = sorted_distances[:, n_neighbors]

    reachability = np.maximum(
        k_distance[order], np.take_along_axis(distances, order, axis=1)
    )
    lrd = 1.0 / np.maximum(reachability.mean(axis=1), 1e-12)
    lof = (lrd[order].mean(axis=1)) / np.maximum(lrd, 1e-12)
    weights = lof - lof.min()
    if weights.max() > 0:
        return weights / weights.max()
    return np.ones(len(points), dtype=float)
