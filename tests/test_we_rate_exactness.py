"""End-to-end exactness check: does WE + recycling recover a KNOWN rate?

Every other WE test here pins an invariant (weight conserved, frontier spawned,
weights equalised). None of them answers the only question that ultimately matters
for kinetics mode: **does the MFPT come out right?** Invariants can all hold while
the physics is wrong -- that is exactly what happened during development (weight was
conserved to 1.00000000 while the rate diverged by 200x).

So this test does what no unit test can fake: it runs the real WESpawner on a 1D
biased random walk whose MFPT can be measured by brute force, and demands agreement.
Cheap dynamics mean the answer arrives in seconds instead of the 12 GPU-hours an MD
validation costs -- the same strategy the WE-MSM literature uses (a 1D biased random
walk against an analytic answer).

Source = x <= 0, target = x >= L, with a downhill drift so crossing is a genuine
rare event that brute force can still reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.spawners.we import WESpawner

L = 10.0
DRIFT = -0.30
NOISE = 1.0
FPW = 4


def _step(x, rng):
    return np.clip(x + DRIFT + NOISE * rng.normal(size=x.shape), 0.0, None)


def _brute_force_mfpt(n_traj=4000, max_steps=60000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_traj)
    done = np.zeros(n_traj, bool)
    t_hit = np.full(n_traj, np.nan)
    for t in range(1, max_steps + 1):
        x[~done] = _step(x[~done], rng)
        hit = (~done) & (x >= L)
        t_hit[hit] = t
        done |= hit
        if done.all():
            break
    return np.nanmean(t_hit), done.mean()


def _we_mfpt(n_walkers=40, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    sp = WESpawner(
        n_bins=[10, 1], min_values=[0.0, -1.0], max_values=[L, 1.0],
        target_per_bin=4, seed=seed,
        recycle_target=[[L, 1e9], [-1.0, 1.0]],
        recycle_basis_index=0,
    )
    pos = np.zeros(n_walkers)
    for _ in range(iters):
        pts = np.column_stack([np.repeat(pos, FPW), np.zeros(n_walkers * FPW)])
        pts[0, 0] = 0.0                       # frame 0 IS the basis (x = 0)
        sp.sample(pts, top_n=n_walkers)
        parents = np.asarray(sp.selected_parents)
        child = np.where(parents < 0, 0.0, pos[np.clip(parents, 0, None)])
        pos = _step(child, rng)
    return sp


def test_we_recycling_recovers_the_brute_force_mfpt():
    """The headline claim of kinetics mode, checked against ground truth."""
    mfpt_bf, frac = _brute_force_mfpt()
    assert frac > 0.99, "brute force must actually reach the target to be a reference"

    sp = _we_mfpt()
    flux = np.asarray(sp.flux_history, float)
    tail = flux[len(flux) // 2:]          # discard the pre-steady-state transient
    mfpt_we = 1.0 / tail.mean()

    # WE is a statistical estimator; demand the right answer within a factor of 1.5.
    # (Development bug produced 200x, so this is a wide but decisive net.)
    assert 0.67 * mfpt_bf < mfpt_we < 1.5 * mfpt_bf, (
        f"WE MFPT {mfpt_we:.0f} vs brute force {mfpt_bf:.0f} steps"
    )


def test_steady_state_flux_plateaus_rather_than_decaying():
    """A decaying flux means weight is draining out of the pipeline -- the failure
    mode that made the MFPT climb without bound (0.13 -> 0.9 -> 1.5 -> 70 ns)."""
    sp = _we_mfpt()
    f = np.asarray(sp.flux_history, float)
    n = len(f)
    q3 = f[2 * n // 4:3 * n // 4].mean()
    q4 = f[3 * n // 4:].mean()
    assert q3 > 0 and q4 > 0
    # late-run flux must not collapse relative to mid-run
    assert 0.5 < q4 / q3 < 2.0, f"flux drifting: Q3={q3:.2e} Q4={q4:.2e}"


def test_walker_weights_do_not_collapse_over_a_long_run():
    """No exponential underflow: min weight fell 3e-5 -> 2.9e-21 before the fix."""
    sp = _we_mfpt()
    w = np.asarray(sp.weights, float)
    assert w.sum() == pytest.approx(1.0)
    assert w.min() > 1e-12, f"walker weights collapsing: min={w.min():.2e}"
