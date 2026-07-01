"""OpenMM system loader for the AIB9 Trails-MD example."""

from __future__ import annotations

from pathlib import Path

from openmm import LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, picoseconds


BASE_DIR = Path(__file__).resolve().parent


def make_system(_topology_source, temp=400.0, dt=0.001):
    """Load the verified AIB9 CHARMM system built by build_aib9.py."""

    with (BASE_DIR / "aib9_system.xml").open() as handle:
        system = XmlSerializer.deserialize(handle.read())
    integrator = LangevinMiddleIntegrator(
        temp * kelvin,
        1.0 / picosecond,
        dt * picoseconds,
    )
    return system, integrator
