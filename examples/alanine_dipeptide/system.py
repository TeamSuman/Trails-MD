"""OpenMM system loader for the alanine-dipeptide hello-world example."""

from __future__ import annotations

from pathlib import Path

from openmm import LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, picoseconds

BASE_DIR = Path(__file__).resolve().parent


def make_system(_topology_source, temp=300.0, dt=0.002):
    """Load the vacuum alanine-dipeptide system built by build_system.py."""
    with (BASE_DIR / "system.xml").open() as handle:
        system = XmlSerializer.deserialize(handle.read())
    integrator = LangevinMiddleIntegrator(
        temp * kelvin,
        1.0 / picosecond,
        dt * picoseconds,
    )
    return system, integrator
