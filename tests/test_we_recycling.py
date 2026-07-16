"""Source->sink recycling: the steady-state rate mode.

A walker reaching the target is terminated and its weight restarted from the basis
(source). That sustains a non-equilibrium steady state whose probability flux into
the target gives MFPT = 1/flux (the Hill relation) -- the same estimator WESTPA
uses, which is what makes the two tools directly comparable.

Everything here guards a number that would otherwise be *plausible and wrong*:
weight must be conserved (never created or destroyed by recycling), the flux must
be booked exactly once per arriving walker, recycled walkers must be re-seeded at
the basis with FRESH velocities (they are new trajectories, not continuations), and
the MFPT must discard the pre-steady-state transient.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.spawners.we import WESpawner

FPW = 10  # frames per walker

# 1-D progress coordinate: source basin near 0, target beyond 90.
SOURCE_BOX = [[-5.0, 10.0], [-1.0, 1.0]]
TARGET_BOX = [[90.0, 200.0], [-1.0, 1.0]]


def _ensemble(progress: np.ndarray) -> np.ndarray:
    flat = np.repeat(progress, FPW)
    return np.column_stack([flat, np.zeros_like(flat)])


def _spawner(seed: int = 0, **kw) -> WESpawner:
    return WESpawner(
        n_bins=[10, 1], min_values=[-5.0, -1.0], max_values=[200.0, 1.0],
        target_per_bin=2, seed=seed, recycle_target=TARGET_BOX, **kw
    )


def test_no_recycling_configured_is_a_no_op():
    """Without recycle_target the spawner must behave exactly as before."""
    sp = WESpawner(n_bins=[10, 1], min_values=[-5.0, -1.0], max_values=[200.0, 1.0],
                   target_per_bin=2, seed=0)
    pts = _ensemble(np.array([1.0, 2.0, 95.0, 3.0]))
    sp.sample(pts, top_n=4)
    assert sp.flux_history == []
    assert all(p >= 0 for p in sp.selected_parents)  # nothing marked recycled


def test_walker_in_target_is_recycled_and_flux_booked():
    sp = _spawner()
    # walker 2 sits in the target; the rest are in the source basin
    progress = np.array([1.0, 2.0, 95.0, 3.0])
    pts = _ensemble(progress)
    sp.sample(pts, top_n=4)

    # exactly one walker arrived, carrying 1/4 of the weight -> flux = 0.25 per tau
    assert len(sp.flux_history) == 1
    assert sp.flux_history[0] == pytest.approx(0.25)


def test_recycling_conserves_weight():
    """Recycling moves weight back to the source -- it never creates or destroys it."""
    sp = _spawner()
    pts = _ensemble(np.array([1.0, 2.0, 95.0, 120.0]))
    for _ in range(5):
        sp.sample(pts, top_n=4)
        assert sp.weights.sum() == pytest.approx(1.0)


def test_recycled_walkers_restart_at_the_basis_with_fresh_velocities():
    """A recycled walker is a NEW trajectory: basis frame + parent marked -1."""
    sp = _spawner(recycle_basis_index=0)
    progress = np.array([1.0, 2.0, 95.0, 3.0])
    pts = _ensemble(progress)
    chosen = sp.sample(pts, top_n=4)

    recycled_slots = [i for i, p in enumerate(sp.selected_parents) if p < 0]
    assert recycled_slots, "the target walker should have produced recycled offspring"
    # -1 tells the orchestrator to draw fresh velocities (not inherit a parent's)
    for i in recycled_slots:
        assert chosen[i] == 0  # restarted from the basis frame


def test_no_flux_when_nothing_reaches_the_target():
    sp = _spawner()
    pts = _ensemble(np.array([1.0, 2.0, 3.0, 4.0]))  # all in the source basin
    sp.sample(pts, top_n=4)
    assert sp.flux_history == [0.0]
    assert sp.mfpt(tau_ps=10.0) is None  # no flux -> no rate, not an infinite one


def test_mfpt_from_flux_discards_the_transient():
    """MFPT = tau/flux, and the pre-steady-state ramp must be dropped.

    The transient systematically under-estimates the flux (the ensemble has not yet
    reached steady state), so averaging over it biases the rate LOW-flux/HIGH-MFPT.
    """
    sp = _spawner()
    # a ramp-up transient (low flux) followed by a steady state at 0.1 per tau
    sp.flux_history = [0.0, 0.0, 0.01, 0.02] + [0.1] * 4
    mfpt = sp.mfpt(tau_ps=10.0, discard_fraction=0.5)
    # steady-state only: 10 ps / 0.1 = 100 ps = 0.1 ns
    assert mfpt == pytest.approx(0.1)
    # including the transient would inflate the MFPT -- confirm it actually differs
    assert sp.mfpt(tau_ps=10.0, discard_fraction=0.0) > mfpt


def test_flux_history_survives_a_checkpoint_roundtrip():
    """The flux series IS the measurement; a resume must not silently lose it."""
    sp = _spawner()
    pts = _ensemble(np.array([1.0, 2.0, 95.0, 3.0]))
    sp.sample(pts, top_n=4)
    state = sp.state_dict()

    fresh = _spawner()
    fresh.load_state_dict(state)
    assert fresh.flux_history == sp.flux_history
