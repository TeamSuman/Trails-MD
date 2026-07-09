#!/usr/bin/env python3
"""Generate a self-contained Amber prmtop/rst7 for the alanine-dipeptide tests.

The repo ships an OpenMM system (``examples/alanine_dipeptide/system.xml`` +
``structure.pdb``) but no Amber topology, so the ``amber_density`` feature in
``run_local_matrix.py`` SKIPs until one exists. This script converts the *same*
Amber14 vacuum parameters into Amber format with **ParmEd**, so the OpenMM and
Amber engines run the identical physical system.

Usage::

    python hpc_tests/assets/build_alad_amber.py            # -> examples/alanine_dipeptide/alad.{prmtop,rst7}
    python hpc_tests/assets/build_alad_amber.py --out-dir /tmp/amber_asset

Requires ``openmm`` and ``parmed`` (both in ``env.yml``). If ParmEd is not
available, a ``tleap`` recipe is printed as a fallback (and written to
``leap.in`` next to the outputs) so the asset can be built with AmberTools:

    tleap -f leap.in   # uses leaprc.protein.ff14SB + structure.pdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EX = REPO_ROOT / "examples" / "alanine_dipeptide"

TLEAP_RECIPE = """\
source leaprc.protein.ff14SB
mol = loadpdb {pdb}
saveamberparm mol {prmtop} {rst7}
quit
"""


def _write_tleap_recipe(out_dir: Path) -> Path:
    leap_in = out_dir / "leap.in"
    leap_in.write_text(
        TLEAP_RECIPE.format(
            pdb=EX / "structure.pdb",
            prmtop=out_dir / "alad.prmtop",
            rst7=out_dir / "alad.rst7",
        )
    )
    return leap_in


def build_with_parmed(out_dir: Path) -> None:
    import parmed as pmd
    from openmm import XmlSerializer
    from openmm.app import PDBFile
    from openmm.unit import angstrom

    pdb = PDBFile(str(EX / "structure.pdb"))
    system = XmlSerializer.deserialize((EX / "system.xml").read_text())

    structure = pmd.openmm.load_topology(pdb.topology, system=system)
    structure.coordinates = pdb.positions.value_in_unit(angstrom)

    prmtop = out_dir / "alad.prmtop"
    rst7 = out_dir / "alad.rst7"
    structure.save(str(prmtop), overwrite=True)
    structure.save(str(rst7), format="rst7", overwrite=True)
    print(f"WROTE {prmtop}")
    print(f"WROTE {rst7}")
    print(f"       {system.getNumParticles()} particles (Amber14 vacuum)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EX,
        help="where to write alad.prmtop/alad.rst7 (default: examples/alanine_dipeptide)",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_with_parmed(args.out_dir)
        return 0
    except ImportError as exc:
        leap_in = _write_tleap_recipe(args.out_dir)
        print(
            f"ParmEd/OpenMM not available ({exc}). Falling back to tleap.",
            file=sys.stderr,
        )
        print(
            f"Wrote a tleap recipe to {leap_in}; build the asset with AmberTools:\n"
            f"    tleap -f {leap_in}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        leap_in = _write_tleap_recipe(args.out_dir)
        print(
            f"ParmEd conversion failed ({type(exc).__name__}: {exc}).", file=sys.stderr
        )
        print(
            f"Use the tleap fallback written to {leap_in}:\n    tleap -f {leap_in}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
