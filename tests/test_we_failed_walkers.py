"""Regression tests: a failed walker must not scramble walker <-> weight <-> endpoint.

When `min_success_fraction < 1` the orchestrator drops walkers whose trajectory failed
and hands the spawner only the SURVIVORS' frames. The weighted-ensemble spawner used to
infer the ensemble size geometrically from the frame count, which is wrong the moment a
walker is missing:

    4 walkers x 10 frames, walker 1 fails -> len(points) = 30
    n_live = min(top_n=4, 30) = 4        <- WRONG, only 3 walkers are live
    fpw    = 30 // 4 = 7                 <- WRONG, segments are 10 frames
    ends   = [6, 13, 20, 27]             <- mid-segment frames, not endpoints;
                                            one survivor counted twice, one lost

and `_live_weights(4)` then saw `len(self.weights) == 4` and MATCHED, silently
re-attaching the previous iteration's weights to the wrong walkers. `sum(w) == 1.0`
still held throughout, so every invariant test passed while the correspondence was
scrambled -- which is precisely why this needs its own test rather than another
invariant check.

The whole path was untested: `grep -rl min_success_fraction tests/` used to return
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.spawners.we import WESpawner

FPW = 10  # frames per walker


def _spawner(n_walkers=4):
    return WESpawner(
        n_bins=[4, 1],
        min_values=[0.0, -1.0],
        max_values=[100.0, 1.0],
        target_per_bin=2,
        seed=0,
    )


def _points(progress: np.ndarray) -> np.ndarray:
    """Each walker contributes FPW frames; the last frame of a block is its endpoint.

    Frames ramp toward the walker's endpoint value so a mid-segment frame is
    numerically distinguishable from the true endpoint.
    """
    rows = []
    for p in progress:
        rows.extend(np.linspace(p - 9.0, p, FPW))
    return np.column_stack([np.asarray(rows), np.zeros(len(rows))])


def test_endpoints_are_endpoints_when_a_walker_fails():
    """THE regression: survivors' true endpoints, not mid-segment frames."""
    sp = _spawner()
    # Walkers 0, 2, 3 survived (walker 1 failed); their endpoints are 20, 40, 50.
    survivors = np.array([20.0, 40.0, 50.0])
    sp.live_walker_indices = [0, 2, 3]
    chosen = sp.sample(_points(survivors), top_n=4)

    # Every parent must name a live walker...
    assert sp.selected_parents is not None
    for parent in sp.selected_parents:
        assert -1 <= parent < len(survivors), (
            f"parent {parent} is outside the live ensemble of {len(survivors)} walkers"
        )
    # ...and the frame each child restarts from must be its parent's TRUE endpoint --
    # the last frame of that parent's block. A bounds check alone was not enough: it
    # passes just as happily when `ends` names block *starts*, so the test called
    # "endpoints are endpoints" did not actually test that. Frames ramp toward the
    # endpoint value, so a mid-segment frame is numerically distinguishable from it.
    for i, parent in enumerate(sp.selected_parents):
        assert chosen[i] == (parent + 1) * FPW - 1, (
            f"child {i} restarts from frame {chosen[i]}, but its parent {parent}'s "
            f"endpoint is frame {(parent + 1) * FPW - 1}"
        )
        assert _points(survivors)[chosen[i], 0] == pytest.approx(survivors[parent]), (
            "the CV at the chosen frame is not the survivor's endpoint CV"
        )


def test_live_weights_follow_the_survivors_not_the_slots():
    """Survivor i must keep walker live_indices[i]'s weight, not walker i's.

    The survivors are deliberately NON-uniform. An earlier draft of this test used
    weights whose survivors happened to renormalise to uniform -- which is exactly
    what the buggy fallback returns, so the test passed against the bug it was
    written to catch. Keep the expected values distinguishable from uniform.
    """
    sp = _spawner()
    sp.weights = np.array([0.4, 0.3, 0.2, 0.1])
    # Walker 1 FAILS. Survivors 0, 2, 3 -> 0.4, 0.2, 0.1 renormalised over 0.7.
    w = sp._live_weights(3, live_indices=[0, 2, 3])

    assert len(w) == 3
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, np.array([0.4, 0.2, 0.1]) / 0.7, rtol=1e-12)
    assert not np.allclose(w, 1 / 3), (
        "expected values must differ from uniform, or this test cannot tell the fix "
        "from the bug's fallback"
    )


def test_live_weights_do_not_shift_onto_the_wrong_walker():
    """The heavy walker survives: it must STAY heavy."""
    sp = _spawner()
    sp.weights = np.array([0.1, 0.7, 0.1, 0.1])
    # Walker 0 fails; survivors 1, 2, 3 -> 0.7, 0.1, 0.1 renormalised over 0.9.
    w = sp._live_weights(3, live_indices=[1, 2, 3])

    np.testing.assert_allclose(w.sum(), 1.0)
    assert w[0] == pytest.approx(0.7 / 0.9), (
        "survivor 0 is previous walker 1, the heavy one; its weight must follow it"
    )


def test_all_walkers_surviving_is_unchanged():
    """The common path must be untouched by the failure handling."""
    sp = _spawner()
    sp.weights = np.array([0.4, 0.3, 0.2, 0.1])
    w = sp._live_weights(4, live_indices=[0, 1, 2, 3])
    np.testing.assert_allclose(w, [0.4, 0.3, 0.2, 0.1])


def test_falls_back_to_uniform_on_a_genuine_count_change():
    """No survivor mapping available -> uniform is the only defensible start."""
    sp = _spawner()
    sp.weights = np.array([0.4, 0.3, 0.2, 0.1])
    w = sp._live_weights(6, live_indices=None)
    np.testing.assert_allclose(w, np.full(6, 1 / 6))


def test_ragged_segments_are_refused_not_silently_misaligned():
    """A short trajectory must raise, not slide every endpoint into a neighbour.

    Endpoints are identified positionally (`block i ends at (i+1)*fpw - 1`), which
    holds only if every live walker contributed the same frame count. Nothing
    upstream enforces that: `build_frame_records` checks only that the counts SUM to
    len(points), and `_validate_trajectory_files` accepts any non-empty file while
    noting a walker "can report success yet leave a truncated file".

    Here walker 0 is truncated to 9 frames. The sum still matches, so the geometric
    guess yields fpw = 29 // 3 = 9 and endpoints [8, 17, 26] -- frame 8 is walker 0's
    real endpoint by luck, but 17 and 26 land mid-segment in the wrong walkers.
    Silent, and fatal to a rate; refuse instead.
    """
    sp = _spawner()
    # 9 + 10 + 10 = 29 frames for 3 live walkers.
    truncated = np.concatenate(
        [np.linspace(11.0, 20.0, 9), np.linspace(31.0, 40.0, 10), np.linspace(41.0, 50.0, 10)]
    )
    points = np.column_stack([truncated, np.zeros(len(truncated))])
    sp.live_walker_indices = [0, 2, 3]

    with pytest.raises(ValueError, match="not a whole number of frames per walker"):
        sp.sample(points, top_n=4)


def test_equal_segments_are_accepted():
    """The guard must not fire on the normal path."""
    sp = _spawner()
    sp.live_walker_indices = [0, 2, 3]
    sp.sample(_points(np.array([20.0, 40.0, 50.0])), top_n=4)  # 30 frames / 3 walkers
    assert sp.selected_parents is not None


def test_discarded_weight_is_reported_not_hidden(caplog):
    """A failure DISCARDS weight; that must be surfaced, with the amount.

    This test used to be `test_weight_is_conserved_across_a_failure`, and it was hollow
    twice over. Its fixture was exactly uniform (`[0.25]*4`), so the survivors
    renormalised to `[1/3, 1/3, 1/3]` -- identical to the buggy uniform fallback it was
    meant to exclude, the same pathology this file's other tests already document. And
    its only assertion, `sp.weights.sum() == 1`, was a tautology of the rescale at the
    end of `_resample_to_budget`.

    It was also misnamed: weight is NOT conserved across a failure. The failed walker's
    weight cannot be honoured -- its trajectory does not exist -- so it is dropped and
    the survivors are renormalised. That perturbs the steady state, which is precisely
    why the loss has to be reported rather than absorbed silently.
    """
    import logging

    sp = _spawner()
    sp.weights = np.array([0.4, 0.3, 0.2, 0.1])  # non-uniform: survivors != uniform
    sp.live_walker_indices = [0, 2, 3]           # walker 1, carrying 0.3, failed

    with caplog.at_level(logging.WARNING):
        sp.sample(_points(np.array([20.0, 40.0, 50.0])), top_n=4)

    assert "1 walker(s) failed" in caplog.text
    assert "0.3" in caplog.text, (
        f"the discarded weight fraction (0.3) must be reported; got: {caplog.text!r}"
    )
