"""Rigorous weighted-ensemble weight bookkeeping.

These pin the properties an unbiased rate estimate depends on. A weight bug does
not crash anything -- it hands back a plausible, badly wrong MFPT -- so each
invariant is asserted directly rather than inferred from behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.spawners.we import WESpawner

FPW = 10  # frames per walker (step // stride)


def _ensemble(n_walkers: int, progress: np.ndarray) -> np.ndarray:
    """Frames laid out contiguously per walker; `progress[i]` is walker i's position."""
    pts = np.repeat(progress, FPW)
    return np.column_stack([pts, np.zeros_like(pts)])


def _spawner(n_bins: int = 4, target: int = 2, seed: int = 0) -> WESpawner:
    return WESpawner(
        n_bins=[n_bins, 1],
        min_values=[0.0, -1.0],
        max_values=[100.0, 1.0],
        target_per_bin=target,
        seed=seed,
    )


def test_weights_sum_to_one_every_iteration():
    sp = _spawner()
    progress = np.array([5.0, 6.0, 30.0, 55.0, 80.0, 7.0, 8.0, 9.0])
    pts = _ensemble(len(progress), progress)
    for _ in range(10):
        chosen = sp.sample(pts, top_n=len(progress))
        assert len(chosen) == len(progress)
        assert sp.weights.sum() == pytest.approx(1.0)
        assert (sp.weights > 0).all()


def test_split_halves_the_parent_weight():
    """A walker alone in its bin, split c ways, must yield c children of w/c."""
    sp = _spawner(n_bins=2, target=4)
    # two bins: 3 walkers low, 1 walker high (the lone frontier walker)
    progress = np.array([5.0, 6.0, 7.0, 90.0])
    pts = _ensemble(4, progress)
    sp.sample(pts, top_n=4)

    total = sp.weights.sum()
    assert total == pytest.approx(1.0)
    # 4 slots over 2 occupied bins -> 2 each. The frontier bin held ONE walker of
    # weight 1/4, split into 2 -> each child carries exactly half of it.
    frontier = sorted(sp.weights)[:2]
    assert frontier[0] == pytest.approx(0.125, rel=1e-9)
    assert frontier[1] == pytest.approx(0.125, rel=1e-9)


def test_frontier_weight_decays_but_frontier_is_still_simulated():
    """The whole point of WE: tiny weight, full CPU share."""
    sp = _spawner(n_bins=4, target=2)
    progress = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 95.0])
    pts = _ensemble(8, progress)
    chosen = sp.sample(pts, top_n=8)

    # the lone frontier walker (index 7) must be spawned despite carrying 1/8 weight
    picked_progress = pts[np.asarray(chosen), 0]
    assert (picked_progress > 90).sum() >= 1


def test_no_bin_is_ever_dropped():
    """n_occ <= n_live <= top_n by construction, so weight is conserved exactly.

    If this ever fails, some bin got zero slots -- and since scarce slots go to the
    sparsest bins, the dropped one is the densest, carrying almost all the weight.
    The resulting MFPT would look plausible and be badly wrong.
    """
    sp = _spawner(n_bins=20, target=1)
    progress = np.linspace(5, 95, 12)   # 12 walkers spread over many bins
    pts = _ensemble(12, progress)
    chosen = sp.sample(pts, top_n=12)
    assert len(chosen) == 12
    assert sp.weights.sum() == pytest.approx(1.0)


def test_weights_are_inherited_not_reset_each_iteration():
    """Weight must follow the lineage; a fresh uniform reset would erase the rate."""
    sp = _spawner(n_bins=2, target=4)
    progress = np.array([5.0, 6.0, 7.0, 90.0])
    pts = _ensemble(4, progress)

    sp.sample(pts, top_n=4)
    first = np.sort(sp.weights.copy())
    sp.sample(pts, top_n=4)
    second = np.sort(sp.weights)

    # Not a uniform reset, and the light (frontier) tail keeps shrinking.
    assert not np.allclose(second, 1.0 / 4)
    assert second[0] <= first[0] + 1e-12
