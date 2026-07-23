"""Regression tests: a recycled walker restarts from the BASIS, and the basis never drifts.

In source->sink kinetics mode a walker that reaches the target is terminated, its weight
booked as flux, and restarted from the basis (source). MFPT = 1/flux, so anything that
corrupts *where a recycled walker actually restarts* corrupts the rate.

The failure these tests exist to catch: the spawner names the restart point by frame
index, but WE does not pool history, so an index only ever addresses the CURRENT
iteration's frames. `recycle_basis_index = 0` therefore means "the basis" at iteration 0
and "wherever walker 0 happens to be now" at iteration N. A recycled walker was booked as
flux and binned at the frozen basis CV while being physically restarted from an unrelated
structure -- and if walker 0 had drifted toward the target, it re-entered the sink almost
immediately and the flux came out inflated (MFPT biased fast). The index cannot carry the
basis; only the structure can.

Note for anyone extending these: the previous tests missed this because the harness
manufactured the invariant. One pinned `pts[0, 0] = 0.0  # frame 0 IS the basis` into
every iteration; another asserted `chosen[i] == 0` against a function that returns the
constant 0. A test must not supply the property it is checking.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.core import TrailsMDCore


class _Core(TrailsMDCore):
    """Bare instance: we exercise only the basis-capture method."""

    def __init__(self, outdir):  # noqa: D107 - deliberately skip the heavy __init__
        self.outdir = outdir
        self._basis_state = None


class _FakeExtractor:
    """Returns a structure that encodes which (iteration, frame) it came from.

    This is the whole point: a real extractor would hand back *some* valid structure
    for frame 0 at any iteration, so a drifting basis is invisible. Here the returned
    positions carry their provenance, so drift is detectable.
    """

    def __init__(self):
        self.iteration = 0

    def extract_positions_by_indices(self, trajectories, indices):
        from openmm import Vec3
        from openmm.unit import nanometer

        out = []
        for index in indices:
            # positions encode (iteration, frame) so we can prove which was used
            positions = np.full((3, 3), float(self.iteration) * 100 + float(index))
            out.append(
                {
                    "positions": positions * nanometer,
                    "box_vectors": tuple(
                        Vec3(*row) * nanometer for row in np.eye(3) * 2.0
                    ),
                }
            )
        return out


def _tag(state) -> float:
    """The (iteration, frame) tag baked into a returned structure."""
    from openmm.unit import nanometer

    return float(np.asarray(state["positions"].value_in_unit(nanometer)).flat[0])


@pytest.fixture
def core(tmp_path):
    return _Core(tmp_path)


def test_basis_is_captured_from_the_first_spawn(core):
    extractor = _FakeExtractor()
    state = core._recycling_basis_state(extractor, ["traj0"], 0)
    assert _tag(state) == 0.0  # iteration 0, frame 0


def test_basis_does_not_drift_across_iterations(core):
    """THE regression. Frame 0 means something different at every iteration."""
    extractor = _FakeExtractor()
    first = _tag(core._recycling_basis_state(extractor, ["traj0"], 0))

    for iteration in range(1, 60):
        extractor.iteration = iteration
        later = _tag(core._recycling_basis_state(extractor, [f"traj{iteration}"], 0))
        assert later == first, (
            f"basis drifted at iteration {iteration}: {later} != {first}. "
            "A recycled walker would restart from a mid-run structure and the "
            "flux -- hence the MFPT -- would be wrong."
        )


def test_basis_survives_a_resume(core, tmp_path):
    """A resumed run must reload the ORIGINAL basis, not re-capture a mid-run frame."""
    extractor = _FakeExtractor()
    first = _tag(core._recycling_basis_state(extractor, ["traj0"], 0))

    # Simulate a restart: fresh core, same outdir, run now well past iteration 0.
    resumed = _Core(tmp_path)
    extractor.iteration = 42
    recovered = _tag(resumed._recycling_basis_state(extractor, ["traj42"], 0))

    assert recovered == first, (
        "after resume the basis was re-captured from the current iteration; "
        "the persisted basis must win."
    )


def test_basis_is_persisted_to_disk(core, tmp_path):
    core._recycling_basis_state(_FakeExtractor(), ["traj0"], 0)
    path = tmp_path / "recycling_basis_state.npz"
    assert path.exists(), "basis must be persisted for resume"
    # np.savez appends .npz to a suffix-less name, which silently breaks the atomic
    # rename; guard against that regression explicitly.
    assert not (tmp_path / "recycling_basis_state.npz.npz").exists()
    assert not list(tmp_path.glob("*.tmp.npz")), "temp file must be renamed away"


def test_box_vectors_round_trip_through_persistence(core, tmp_path):
    from openmm.unit import nanometer

    original = core._recycling_basis_state(_FakeExtractor(), ["traj0"], 0)
    box_before = np.array(
        [list(v.value_in_unit(nanometer)) for v in original["box_vectors"]]
    )

    resumed = _Core(tmp_path)
    reloaded = resumed._recycling_basis_state(_FakeExtractor(), ["traj0"], 0)
    box_after = np.array(
        [list(v.value_in_unit(nanometer)) for v in reloaded["box_vectors"]]
    )

    np.testing.assert_allclose(box_after, box_before)


def test_missing_box_vectors_round_trip(core, tmp_path):
    """A system with no periodic box must reload as None, not as an empty tuple."""
    from openmm.unit import nanometer

    class _NoBox(_FakeExtractor):
        def extract_positions_by_indices(self, trajectories, indices):
            states = super().extract_positions_by_indices(trajectories, indices)
            for s in states:
                s["box_vectors"] = None
            return states

    core._recycling_basis_state(_NoBox(), ["traj0"], 0)
    resumed = _Core(tmp_path)
    reloaded = resumed._recycling_basis_state(_NoBox(), ["traj0"], 0)
    assert reloaded["box_vectors"] is None
    assert np.asarray(reloaded["positions"].value_in_unit(nanometer)).shape == (3, 3)


def test_desynced_basis_cv_and_structure_are_refused(core):
    """A restored basis_cv with no .npz must raise, not re-capture from a live frame.

    The frozen CV (spawner checkpoint) and the frozen structure (.npz) are two
    independent persistence paths, and only their AGREEMENT makes the pair meaningful:
    the CV is where a recycled walker is binned, the structure is where it actually
    restarts. Resuming a run whose .npz went missing restores the CV but re-captures
    the structure from whatever frame `basis_index` names *now* -- silently
    reinstating the drift, because a perfectly plausible structure still comes back.
    """
    extractor = _FakeExtractor()
    extractor.iteration = 25  # resumed mid-run: frame 0 is no longer the basis

    with pytest.raises(RuntimeError, match="Recycling basis is inconsistent"):
        core._recycling_basis_state(
            extractor,
            ["traj25"],
            0,
            expected_cv=np.array([0.0, 0.0]),   # frozen at the source, from checkpoint
            observed_cv=np.array([8.4, 1.2]),   # where frame 0 sits at iteration 25
        )


def test_consistent_basis_cv_and_structure_are_accepted(core):
    """The fresh-run path: the CV and the frame agree, so capture proceeds."""
    extractor = _FakeExtractor()
    state = core._recycling_basis_state(
        extractor,
        ["traj0"],
        0,
        expected_cv=np.array([0.0, 0.0]),
        observed_cv=np.array([0.0, 0.0]),
    )
    assert _tag(state) == 0.0


def test_check_is_skipped_once_the_basis_is_already_held(core):
    """A cached/reloaded basis must not be re-validated against a live frame.

    After the basis is captured (or reloaded from the .npz) it is returned as-is, so a
    drifted current frame is expected and must NOT raise -- otherwise every run would
    die at iteration 2, when frame 0 has legitimately moved on.
    """
    extractor = _FakeExtractor()
    core._recycling_basis_state(extractor, ["traj0"], 0)  # capture

    extractor.iteration = 30
    state = core._recycling_basis_state(
        extractor,
        ["traj30"],
        0,
        expected_cv=np.array([0.0, 0.0]),
        observed_cv=np.array([9.9, 9.9]),  # wildly drifted -- irrelevant, basis is held
    )
    assert _tag(state) == 0.0
