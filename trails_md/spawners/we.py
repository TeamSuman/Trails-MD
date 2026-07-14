"""Weighted-ensemble spawner.

Carries statistical weights on the cumulative point cloud and uses
:class:`~trails_md.binning.we.WeightedEnsemble` split/merge resampling to pick
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

from trails_md.binning.spatial import RegularBinner
from trails_md.binning.we import WeightedEnsemble

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
        **kwargs: Any,
    ):
        super().__init__(seed=seed, **kwargs)
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.we = WeightedEnsemble(target_per_bin=target_per_bin)
        self.weights: np.ndarray | None = None  # aligned to cumulative cloud
        # Optional landscape-adaptive binner (set by the orchestrator); None -> grid.
        self.binner = None

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
        result = self.we.resample(weights, labels, rng=self.rng)

        # Carry weights forward: aggregate resampled weight onto parent frames.
        new_weights = np.zeros(n, dtype=float)
        for parent, w in zip(result.parents, result.weights, strict=False):
            new_weights[parent] += w
        total = new_weights.sum()
        self.weights = new_weights / total if total > 0 else None

        return self._draw(result, labels, top_n, self.rng)

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
        if self.binner is not None:
            self.binner.n_bins = np.asarray(self.n_bins, dtype=int)
            table = self.binner.fit(cumulative)
        else:
            table = RegularBinner(
                n_bins=self.n_bins,
                min_values=self.min_values,
                max_values=self.max_values,
            ).fit(cumulative)
        labels = np.full(len(cumulative), -1, dtype=int)
        for row, frames in enumerate(table.populated_data):
            for frame in frames:
                labels[frame] = row
        return labels

    @staticmethod
    def _draw(
        result, labels: np.ndarray, top_n: int, rng: np.random.Generator
    ) -> list[int]:
        """Allocate the walker budget *across bins*, never in proportion to weight.

        This is the heart of weighted ensemble and the easiest thing to get
        backwards. A walker sitting on top of a barrier carries an
        exponentially small statistical weight -- that smallness is the
        *result* being computed, not a reason to stop simulating it. Drawing
        walkers with ``p = weight`` (as this function used to) hands essentially
        the entire budget to the equilibrium basin and reduces WE to plain
        unbiased MD: on the proline barrier the frontier bin held total weight
        4e-6, so it was picked once per ~15,900 iterations, i.e. never in a
        250-iteration run.

        Instead every occupied bin is entitled to an equal share of the budget.
        When there are fewer slots than bins the sparsest bins win the ties --
        the sparsest bin *is* the frontier -- so the leading edge is always
        simulated. Weights are still carried (and still conserved) by
        ``resample``; they are for unbiased estimation, not for CPU allocation.
        """
        parents = np.asarray(result.parents, dtype=int)
        if top_n <= 0 or parents.size == 0:
            return []
        parent_bins = labels[parents]
        bins = np.unique(parent_bins)

        # Bin population over the cumulative cloud: ascending == frontier first.
        pops = np.bincount(labels[labels >= 0], minlength=int(labels.max()) + 1)
        order = np.argsort(pops[bins], kind="stable")

        # Round-robin the slots, sparsest bin first. If slots >= bins every bin
        # is served; if slots < bins the sparsest `top_n` bins are served.
        slots = np.zeros(len(bins), dtype=int)
        for k in range(top_n):
            slots[order[k % len(bins)]] += 1

        chosen: list[int] = []
        for position, count in enumerate(slots):
            if count == 0:
                continue
            candidates = parents[parent_bins == bins[position]]
            # Walkers within a bin are interchangeable after split/merge.
            picks = rng.choice(len(candidates), size=count, replace=len(candidates) < count)
            chosen.extend(int(candidates[p]) for p in picks)
        return chosen

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["weights"] = None if self.weights is None else self.weights.tolist()
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if state and state.get("weights") is not None:
            self.weights = np.asarray(state["weights"], dtype=float)



SpawnerFactory.register("we", WESpawner)
