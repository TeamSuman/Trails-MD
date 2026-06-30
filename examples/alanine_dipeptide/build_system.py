"""Generate the vacuum alanine-dipeptide OpenMM system for the hello-world example.

Builds an Amber14 vacuum ``System`` from ``structure.pdb`` (Ac-Ala-NMe, 22 atoms)
and serialises it to ``system.xml``. Run once to (re)generate the committed asset:

    python examples/alanine_dipeptide/build_system.py

``structure.pdb`` is the canonical alanine-dipeptide test structure from the
OpenMM project (MIT licensed). Only OpenMM is required — no external force-field
files, no GPU.
"""

from __future__ import annotations

from pathlib import Path

from openmm import XmlSerializer
from openmm.app import ForceField, NoCutoff, PDBFile

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    pdb = PDBFile(str(BASE_DIR / "structure.pdb"))
    forcefield = ForceField("amber14-all.xml")  # vacuum: no water model
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=NoCutoff, constraints=None
    )
    out = BASE_DIR / "system.xml"
    out.write_text(XmlSerializer.serialize(system))
    print("WROTE", out, "with", system.getNumParticles(), "particles")


if __name__ == "__main__":
    main()
