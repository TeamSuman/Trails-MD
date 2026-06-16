from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from autosampler.binning.spatial import BinTable, RegularBinner
from .base import Spawner, SpawnerFactory


class DensitySpawner(Spawner):
    """Selects frames from sparsely populated regular grid bins."""

    def __init__(
        self,
        n_bins: list[int] | None = None,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        mode: str = "explore",
        probabilistic: bool = True,
        target: list[float] | None = None,
        recent_window: int = 5,
        **_: Any,
    ):
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.mode = mode
        self.probabilistic = probabilistic
        self.target = target
        self.recent_bins: deque[set[Any]] = deque(maxlen=recent_window)

    def sample(self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None) -> list[int]:
        points = np.asarray(points, dtype=float)
        cumulative_points = _cumulative_points(points, history)
        binner = RegularBinner(
            n_bins=self.n_bins,
            min_values=self.min_values,
            max_values=self.max_values,
            target=self.target if self.mode == "target" else None,
        )
        table = binner.fit(cumulative_points)
        selected_rows = (
            self._probabilistic_rows(table, top_n)
            if self.probabilistic
            else self._hard_rows(table, top_n)
        )
        if not self.probabilistic:
            self.recent_bins.append({table.ids[row] for row in selected_rows})
        return _sample_frames(table, selected_rows)

    def _probabilistic_rows(self, table: BinTable, top_n: int) -> np.ndarray:
        occupied = table.occupied_indices
        if len(occupied) == 0:
            raise ValueError("Cannot sample density bins: no populated bins found.")
        weights = np.zeros(len(table.ids), dtype=float)
        weights[occupied] = 1.0 / table.populations[occupied]
        if self.mode == "target":
            if table.target_closeness is None:
                raise ValueError("Target density sampling requires a target.")
            weights *= table.target_closeness
        return _weighted_choice(occupied, weights[occupied], top_n)

    def _hard_rows(self, table: BinTable, top_n: int) -> np.ndarray:
        occupied = table.occupied_indices
        if len(occupied) == 0:
            raise ValueError("Cannot sample density bins: no populated bins found.")

        recent = set().union(*self.recent_bins) if self.recent_bins else set()
        selected: list[int] = []
        for population in sorted(np.unique(table.populations[occupied])):
            rows = occupied[table.populations[occupied] == population]
            if self.mode == "target" and table.target_closeness is not None:
                rows = rows[np.argsort(table.target_closeness[rows])[::-1]]
            fresh = [row for row in rows if table.ids[int(row)] not in recent]
            stale = [row for row in rows if table.ids[int(row)] in recent]
            ordered = list(np.random.permutation(fresh)) + list(np.random.permutation(stale))
            selected.extend(int(row) for row in ordered)
            if len(selected) >= top_n:
                return np.asarray(selected[:top_n], dtype=int)

        repeats = int(np.ceil(top_n / len(selected)))
        return np.tile(np.asarray(selected, dtype=int), repeats)[:top_n]


def _sample_frames(table: BinTable, rows: np.ndarray) -> list[int]:
    return [int(np.random.choice(table.populated_data[int(row)])) for row in rows]


def _cumulative_points(points: np.ndarray, history: dict[int, Any] | None) -> np.ndarray:
    historical = _historical_points(points, history)
    if historical.size == 0:
        return points
    return np.vstack([historical, points])


def _historical_points(points: np.ndarray, history: dict[int, Any] | None) -> np.ndarray:
    if not history:
        return np.empty((0, points.shape[1]), dtype=float)

    projections = []
    for iteration in sorted(history):
        entry = history[iteration]
        projection = entry.get("projection") if isinstance(entry, dict) else None
        if projection is None:
            continue
        projection = np.asarray(projection, dtype=float)
        if projection.ndim == 2 and projection.shape[1] == points.shape[1]:
            projections.append(projection)

    return np.vstack(projections) if projections else np.empty((0, points.shape[1]), dtype=float)


def _weighted_choice(rows: np.ndarray, weights: np.ndarray, top_n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    weights = weights / weights.sum()
    replace = len(rows) < top_n
    return np.random.choice(rows, size=top_n, replace=replace, p=weights).astype(int)


SpawnerFactory.register("density", DensitySpawner)
