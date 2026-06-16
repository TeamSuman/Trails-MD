from __future__ import annotations

from typing import Any

import numpy as np

from autosampler.binning.spatial import VoronoiBinner
from .base import SpawnerFactory
from .density import DensitySpawner, _cumulative_points, _sample_frames, _weighted_choice


class VoronoiSpawner(DensitySpawner):
    """Samples frames from large, sparsely populated Voronoi cells."""

    def __init__(
        self,
        n_clusters: int = 150,
        mode: str = "explore",
        target: list[float] | None = None,
        periodic: bool = False,
        grid_size: int = 250,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__(mode=mode, target=target, **kwargs)
        self.n_clusters = n_clusters
        self.periodic = periodic
        self.grid_size = grid_size
        self.min_values = min_values
        self.max_values = max_values

    def sample(self, points: np.ndarray, top_n: int, history=None) -> list[int]:
        points = np.asarray(points, dtype=float)
        cumulative_points = _cumulative_points(points, history)
        binner = VoronoiBinner(
            n_clusters=self.n_clusters,
            min_values=self.min_values,
            max_values=self.max_values,
            target=self.target if self.mode == "target" else None,
            periodic=self.periodic,
            grid_size=self.grid_size,
        )
        table = binner.fit(cumulative_points)
        occupied = table.occupied_indices
        if len(occupied) == 0:
            raise ValueError("Cannot sample Voronoi bins: no populated bins found.")

        weights = table.area[occupied] / table.populations[occupied]
        if self.mode == "target":
            if table.target_closeness is None:
                raise ValueError("Target Voronoi sampling requires a target.")
            weights *= table.target_closeness[occupied]
        rows = _weighted_choice(occupied, weights, top_n)
        return _sample_frames(table, rows)


SpawnerFactory.register("voronoi", VoronoiSpawner)
