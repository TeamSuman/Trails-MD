"""The configured constraint tolerance must actually reach the integrator.

`OpenMMEngine.prepare` sets `self.constraintTolerance = 1e-6`, ten times tighter
than OpenMM's 1e-5 default. That assignment does nothing on its own -- the value
only takes effect via `integrator.setConstraintTolerance`, which is a single line
at the very end of `prepare`.

That line was once left stranded *after* a `return` when a new method was spliced
in above it, making it unreachable: the attribute was still assigned, every test
still passed, `ruff` still reported clean (it has no unreachable-code rule on by
default), and every simulation silently ran at the 1e-5 default instead. Nothing
observable failed -- constraint tolerance does not crash, it just quietly changes
the physics, and it would have differed between benchmark runs for a reason that
had nothing to do with what was being benchmarked.

So assert the integrator's ACTUAL state, not the attribute: reading back
`self.constraintTolerance` would pass against exactly that bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openmm")

from trails_md.engines.openmm import OpenMMEngine  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alanine_dipeptide"
CONFIGURED_TOLERANCE = 1e-6
OPENMM_DEFAULT_TOLERANCE = 1e-5


def _prepared_engine() -> OpenMMEngine:
    engine = OpenMMEngine(platform_name="CPU", dt=0.002, seed=7)
    engine.prepare(
        conf=EXAMPLE / "structure.pdb",
        top=EXAMPLE / "structure.pdb",
        system_file=EXAMPLE / "system.py",
    )
    return engine


def test_constraint_tolerance_reaches_the_integrator():
    """THE regression: the setter must run, not merely the attribute assignment."""
    engine = _prepared_engine()

    assert engine.integrator.getConstraintTolerance() == pytest.approx(
        CONFIGURED_TOLERANCE
    ), (
        "the integrator is not using the configured constraint tolerance -- the "
        "setConstraintTolerance call in prepare() is unreachable or missing"
    )


def test_configured_tolerance_differs_from_the_openmm_default():
    """Guards the test above: if the two ever coincide, it proves nothing.

    The regression this file exists for left the integrator at OpenMM's default.
    If the configured value were ever changed to equal that default, the assertion
    above would pass whether or not the setter ran -- hollow, in exactly the way
    that let the original bug through.
    """
    engine = _prepared_engine()

    assert engine.constraintTolerance != OPENMM_DEFAULT_TOLERANCE, (
        "configured tolerance now equals OpenMM's default, so the reachability "
        "test above can no longer detect a skipped setConstraintTolerance call"
    )
    assert engine.integrator.getConstraintTolerance() != pytest.approx(
        OPENMM_DEFAULT_TOLERANCE
    )
