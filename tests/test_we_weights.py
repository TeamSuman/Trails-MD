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


def test_resampling_equalises_weights_no_zombie_walkers():
    """Within a bin, resampling must EQUALISE weights -- not just fix the count.

    Getting only the count right lets light walkers merge with each other forever,
    never absorbed into the heavy ones. They survive as "zombies" carrying ~1e-21:
    they hold a walker slot, cost a full MD segment, and contribute nothing. On a
    real alanine run this collapsed the effective sample size to 3.6 of 40 walkers
    and made the flux (hence the rate) decay without bound.
    """
    from trails_md.spawners.we import WESpawner

    # one bin, one heavy walker + four zombies -- the exact pathology observed
    members = [0, 1, 2, 3, 4]
    mweights = [0.4, 1e-21, 1e-21, 1e-21, 1e-21]
    out_m, out_w = WESpawner._resample_bin(
        members, mweights, target=4, rng=np.random.default_rng(0)
    )
    out_w = np.asarray(out_w)

    assert len(out_m) == 4                                   # count respected
    assert out_w.sum() == pytest.approx(0.4)                 # weight conserved
    assert out_w.max() <= 4.0 * out_w.min()                  # weights equalised
    assert (out_w > 1e-6).all()                              # no zombies survive
    # effective sample size should be ~the walker count, not ~1
    ess = 1.0 / np.sum((out_w / out_w.sum()) ** 2)
    assert ess > 3.0


def test_within_bin_weights_stay_equal_over_many_iterations():
    """A proper WE random walk: walkers MOVE and their weights follow them.

    This is the honest version of the steady-state check. Positions must track the
    resampled walkers (children inherit the parent's position), or the test decouples
    weights from positions and measures nothing. Across bins, WE weights are meant to
    be wildly unequal -- that is how it samples improbable regions. What must NOT
    happen is unequal weights WITHIN a bin: that is the zombie pathology.
    """
    from trails_md.spawners.we import WESpawner

    fpw = 10
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 10.0, 24)          # all walkers start in the source basin
    sp = WESpawner(n_bins=[6, 1], min_values=[0.0, -1.0], max_values=[60.0, 1.0],
                   target_per_bin=4, seed=0)

    for _ in range(60):
        pts = np.column_stack([np.repeat(pos, fpw), np.zeros(len(pos) * fpw)])
        sp.sample(pts, top_n=len(pos))
        # children inherit the parent's position, then diffuse (biased slightly uphill)
        parents = np.asarray(sp.selected_parents)
        pos = pos[parents] + rng.normal(0.4, 1.5, len(parents))
        pos = np.clip(pos, 0.0, 60.0)

    w = np.asarray(sp.weights, float)
    assert w.sum() == pytest.approx(1.0)
    # WITHIN each bin the weights must be comparable -- no zombie slots
    b = np.clip((pos / 10.0).astype(int), 0, 5)
    for bin_id in np.unique(b):
        m = b == bin_id
        if m.sum() > 1:
            wb = w[m]
            assert wb.max() <= 1e3 * wb.min(), (
                f"bin {bin_id}: within-bin weight spread {wb.max()/wb.min():.1e} "
                f"-- zombie walkers holding slots but no weight"
            )
