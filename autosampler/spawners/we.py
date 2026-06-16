"""Weighted-ensemble spawner.

Carries statistical weights on the cumulative point cloud and uses
:class:`~autosampler.binning.we.WeightedEnsemble` split/merge resampling to pick
the next walkers, conserving total weight. A faithful alternative to MSM
least-counts / density spawning when unbiased weights are wanted.

Implements the standard ``sample(points, top_n, history)`` contract, returning
indices into the cumulative point cloud (repeats indicate split walkers), so it
is a drop-in ``spawn_scheme: we`` option. Per-frame weights are carried across
iterations in the spawner instance (and exposed via ``state_dict``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autosampler.binning.spatial import RegularBinner
from autosampler.binning.we import WeightedEnsemble

from .base import Spawner, SpawnerFactory
from .density import _cumulative_points


class WESpawner(Spawner):
    def __init__(
        self,
        n_bins: list[int] | None = None,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        target_per_bin: int = 4,
        seed: int = 42,
        **_: Any,
    ):
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.we = WeightedEnsemble(target_per_bin=target_per_bin)
        self.seed = int(seed)
        self.weights: np.ndarray | None = None  # aligned to cumulative cloud

    def sample(
        self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None
    ) -> list[int]:
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            raise ValueError("Cannot run WE on an empty point cloud.")
        cumulative = _cumulative_points(points, history)
        n = len(cumulative)

        # Initialise / extend per-frame weights; new frames enter with the mean
        # weight so they neither dominate nor vanish, then renormalise to 1.
        weights = self._extend_weights(n)

        labels = self._bin_labels(cumulative)
        rng = np.random.default_rng(self.seed)
        result = self.we.resample(weights, labels, rng=rng)

        # Carry weights forward: aggregate resampled weight onto parent frames.
        new_weights = np.zeros(n, dtype=float)
        for parent, w in zip(result.parents, result.weights, strict=False):
            new_weights[parent] += w
        total = new_weights.sum()
        self.weights = new_weights / total if total > 0 else None

        return self._draw(result, top_n, rng)

    def _extend_weights(self, n: int) -> np.ndarray:
        if self.weights is None or len(self.weights) == 0:
            weights = np.full(n, 1.0 / n, dtype=float)
        elif len(self.weights) < n:
            fill = float(np.mean(self.weights)) if len(self.weights) else 1.0
            weights = np.concatenate(
                [self.weights, np.full(n - len(self.weights), fill)]
            )
        else:
            weights = np.asarray(self.weights[:n], dtype=float)
        total = weights.sum()
        return weights / total if total > 0 else np.full(n, 1.0 / n)

    def _bin_labels(self, cumulative: np.ndarray) -> np.ndarray:
        binner = RegularBinner(
            n_bins=self.n_bins, min_values=self.min_values, max_values=self.max_values
        )
        table = binner.fit(cumulative)
        labels = np.full(len(cumulative), -1, dtype=int)
        for row, frames in enumerate(table.populated_data):
            for frame in frames:
                labels[frame] = row
        return labels

    @staticmethod
    def _draw(result, top_n: int, rng: np.random.Generator) -> list[int]:
        parents = np.asarray(result.parents, dtype=int)
        weights = np.asarray(result.weights, dtype=float)
        total = weights.sum()
        probs = weights / total if total > 0 else np.full(len(parents), 1.0 / len(parents))
        replace = len(parents) < top_n
        chosen = rng.choice(parents, size=top_n, replace=replace, p=probs)
        return [int(i) for i in chosen]

    def state_dict(self) -> dict:
        return {"weights": None if self.weights is None else self.weights.tolist()}

    def load_state_dict(self, state: dict) -> None:
        if state and state.get("weights") is not None:
            self.weights = np.asarray(state["weights"], dtype=float)


SpawnerFactory.register("we", WESpawner)
