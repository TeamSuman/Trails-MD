"""Regression tests for the WE frontier-starvation bug.

`WESpawner._draw` used to pick the next walkers with probability proportional to
their statistical weight. That inverts the point of weighted ensemble: a walker
on top of a barrier carries an exponentially small weight *by construction*, so
weight-proportional selection spends the whole budget on the equilibrium basin
and WE silently degenerates into unbiased MD.

Measured on the real explicit-solvent proline barrier (frames per 10 deg of
progress: 176124 / 67107 / 10353 / 775 / 40 / 1) the frontier bin held total
weight 3.9e-6, so it was selected once per ~15,900 iterations -- never, in a
250-iteration campaign. That is exactly why `spawn_scheme: we` scored the same
frontier angle as plain unbiased MD.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from trails_md.binning.we import WeightedEnsemble
from trails_md.spawners.we import WESpawner

# Frames per 10-degree slice up the real proline barrier (explicit TIP3P).
PROLINE_POPULATIONS = [176124, 67107, 10353, 775, 40, 1]


def _proline_cloud() -> np.ndarray:
    """A 2-D cloud whose first axis is progress up the barrier (0..50 deg)."""
    progress = np.concatenate(
        [np.full(pop, 10.0 * i) for i, pop in enumerate(PROLINE_POPULATIONS)]
    )
    return np.column_stack([progress, np.zeros_like(progress)])


def _spawner(seed: int = 0) -> WESpawner:
    return WESpawner(
        n_bins=[len(PROLINE_POPULATIONS), 1],
        min_values=[-5.0, -1.0],
        max_values=[10.0 * len(PROLINE_POPULATIONS) - 5.0, 1.0],
        target_per_bin=4,
        seed=seed,
    )


def test_frontier_is_spawned_every_iteration():
    """The sparsest bin -- the barrier frontier -- must always get walkers."""
    points = _proline_cloud()
    frontier = 10.0 * (len(PROLINE_POPULATIONS) - 1)
    spawner = _spawner()

    attacked = 0
    trials = 25
    for _ in range(trials):
        chosen = spawner.sample(points, top_n=16)
        if (points[np.asarray(chosen), 0] >= frontier).any():
            attacked += 1

    # Weight-proportional selection scored ~0.0001 here. Bin-balanced scores 1.0.
    assert attacked == trials


def test_budget_is_spread_across_bins_not_concentrated_in_the_basin():
    """No single bin may swallow the walker budget, however heavy it is."""
    points = _proline_cloud()
    chosen = _spawner().sample(points, top_n=16)
    progress = points[np.asarray(chosen), 0]

    # 6 occupied bins, 16 slots -> every bin served, none monopolising.
    assert len(np.unique(progress)) == len(PROLINE_POPULATIONS)
    _, counts = np.unique(progress, return_counts=True)
    assert counts.max() <= 4


def test_scarce_slots_go_to_the_sparsest_bins():
    """With fewer slots than bins, the frontier still wins a slot."""
    points = _proline_cloud()
    chosen = _spawner().sample(points, top_n=2)
    progress = sorted(points[np.asarray(chosen), 0])
    # The two sparsest bins are the two highest-progress ones.
    assert progress == [40.0, 50.0]


def test_merge_conserves_weight_and_is_not_quadratic():
    """A real equilibrium bin (176k frames) must merge in reasonable time."""
    n = len(PROLINE_POPULATIONS[0:1]) and PROLINE_POPULATIONS[0]
    weights = np.full(n, 1.0 / n)
    labels = np.zeros(n, dtype=int)

    start = time.perf_counter()
    result = WeightedEnsemble(target_per_bin=4).resample(
        weights, labels, rng=np.random.default_rng(0)
    )
    elapsed = time.perf_counter() - start

    assert len(result) == 4
    assert sum(result.weights) == pytest.approx(1.0)
    # The old list-rebuilding merge was O(n^2 log n) and never returned here.
    assert elapsed < 15.0


def test_weights_are_still_conserved_after_resampling():
    """Selection changed; the WE weight bookkeeping must not have."""
    points = _proline_cloud()
    spawner = _spawner()
    spawner.sample(points, top_n=16)
    assert spawner.weights is not None
    assert spawner.weights.sum() == pytest.approx(1.0)
