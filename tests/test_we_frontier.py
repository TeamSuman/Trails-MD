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

FPW = 10  # frames per walker (step // stride)


def _ensemble(progress: np.ndarray) -> np.ndarray:
    """Live ensemble: frames laid out contiguously per walker, one block each."""
    flat = np.repeat(progress, FPW)
    return np.column_stack([flat, np.zeros_like(flat)])


def _spawner(n_bins: int = 6, target: int = 4, seed: int = 0) -> WESpawner:
    return WESpawner(
        n_bins=[n_bins, 1],
        min_values=[-5.0, -1.0],
        max_values=[105.0, 1.0],
        target_per_bin=target,
        seed=seed,
    )


# One lone walker on the barrier top; the rest deep in the trans basin -- the
# shape of the real proline ensemble, where the frontier bin held weight 4e-6.
FRONTIER = 95.0
LIVE = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0, 40.0, 60.0, FRONTIER])


def test_frontier_is_spawned_every_iteration():
    """The sparsest bin -- the barrier frontier -- must always get walkers."""
    points = _ensemble(LIVE)
    spawner = _spawner()

    trials = 25
    attacked = 0
    for _ in range(trials):
        chosen = spawner.sample(points, top_n=len(LIVE))
        if (points[np.asarray(chosen), 0] >= FRONTIER).any():
            attacked += 1

    # Weight-proportional selection scored ~1e-4 here. Bin-balanced scores 1.0.
    assert attacked == trials


def test_budget_is_spread_across_bins_not_concentrated_in_the_basin():
    """No single bin may swallow the walker budget, however heavy it is."""
    points = _ensemble(LIVE)
    spawner = _spawner()
    chosen = spawner.sample(points, top_n=len(LIVE))

    assert len(chosen) == len(LIVE)
    # The crowded trans basin must not take every slot.
    picked = points[np.asarray(chosen), 0]
    assert (picked > 50).sum() >= 2


def test_scarce_slots_go_to_the_sparsest_bins():
    """With fewer slots than occupied bins, the frontier still wins one."""
    points = _ensemble(LIVE)
    chosen = _spawner().sample(points, top_n=2)
    picked = sorted(points[np.asarray(chosen), 0])
    assert len(picked) == 2
    assert picked[-1] == FRONTIER  # the sparsest bin is served first


def test_merge_conserves_weight_and_is_not_quadratic():
    """A real equilibrium bin (176k frames) must merge in reasonable time."""
    n = 176_124
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
    points = _ensemble(LIVE)
    spawner = _spawner()
    spawner.sample(points, top_n=len(LIVE))
    assert spawner.weights is not None
    assert spawner.weights.sum() == pytest.approx(1.0)
