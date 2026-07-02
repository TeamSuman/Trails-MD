"""Phase 0 go/no-go gate: reference-state descriptor calibration (mW).

Per docs/ice_nucleation_cv_protocol.md Phase 0, this script confirms that the
fixed CV (n_max, chi) cleanly separates three reference ensembles:

  1. liquid / disordered   -- a randomized-position negative control
  2. bulk ice Ih           -- IceCoder's shipped seeded-mW crystallization
                               trajectory (data/seeded_mW_scaled.gro +
                               simulation_mW_long_scaled.dcd), which is
                               documented as an Ice-Ih seed growing in a
                               supercooled liquid
  3. bulk ice Ic           -- a synthetic diamond-cubic oxygen lattice built
                               with ASE, at the O-O nearest-neighbor distance
                               measured empirically from (2), so it is not an
                               independently-sourced/guessed lattice constant

Run: python3 phase0_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_descriptors as ic  # noqa: E402

ICECODER_DATA = Path("/home/user/IceCoder/data")
GRO = ICECODER_DATA / "seeded_mW_scaled.gro"
DCD = ICECODER_DATA / "simulation_mW_long_scaled.dcd"


def liquid_reference(n_atoms: int, box_dimensions: np.ndarray, n_seeds: int = 3) -> list[dict]:
    """Randomized-position negative control: destroys all crystal structure."""
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        positions = rng.uniform(0, 1, size=(n_atoms, 3)) * box_dimensions[:3]
        res = ic.ice_cv(positions, box_dimensions)
        results.append(
            {
                "seed": seed,
                "n_max": res.n_max,
                "chi": res.chi,
                "n_ih": res.n_ih,
                "n_ic": res.n_ic,
                "n_interfacial_ice": res.n_interfacial_ice,
                "n_liquid": res.n_liquid,
            }
        )
    return results


def ih_reference_and_nn_distance() -> tuple[list[dict], float]:
    """Ih trajectory sweep + empirical first-shell O-O NN distance."""
    import MDAnalysis as mda
    from freud.locality import AABBQuery

    u = mda.Universe(str(GRO), str(DCD))
    ow = ic.select_water_oxygens(u)
    n_frames = len(u.trajectory)
    frame_indices = sorted(set([0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]))

    results = []
    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        res = ic.ice_cv(ow.positions, u.trajectory.ts.dimensions)
        results.append(
            {
                "frame": frame_idx,
                "n_max": res.n_max,
                "chi": res.chi,
                "n_ih": res.n_ih,
                "n_ic": res.n_ic,
                "n_interfacial_ice": res.n_interfacial_ice,
                "n_liquid": res.n_liquid,
            }
        )

    # Empirical O-O nearest-neighbor distance from the fully-grown crystal's
    # ice-like subset (last frame), used to parameterize the synthetic Ic
    # lattice below without hardcoding a literature mW lattice constant.
    u.trajectory[n_frames - 1]
    chill = ic.chill_plus_labels(ow.positions, u.trajectory.ts.dimensions)
    ice_pos = ow.positions[chill.ice_like]
    fbox = ic._freud_box(u.trajectory.ts.dimensions)
    aq = AABBQuery(fbox, ice_pos)
    nlist = aq.query(ice_pos, {"r_max": 4.0, "exclude_ii": True}).toNeighborList()
    dists = nlist.distances
    first_shell = dists[dists < 3.2]
    nn_distance = float(first_shell.mean())
    return results, nn_distance


def ic_reference(nn_distance: float) -> dict:
    """Synthetic bulk Ic (diamond-cubic) lattice at the measured NN distance."""
    from ase.lattice.cubic import Diamond

    lattice_constant = 4.0 * nn_distance / np.sqrt(3.0)
    atoms = Diamond(symbol="O", latticeconstant=lattice_constant, size=(6, 6, 6))
    positions = atoms.get_positions()
    cell = atoms.cell.array
    dims = np.array([cell[0, 0], cell[1, 1], cell[2, 2], 90.0, 90.0, 90.0])
    res = ic.ice_cv(positions, dims)
    return {
        "lattice_constant": lattice_constant,
        "n_atoms": len(positions),
        "n_max": res.n_max,
        "chi": res.chi,
        "n_ih": res.n_ih,
        "n_ic": res.n_ic,
        "n_interfacial_ice": res.n_interfacial_ice,
        "n_liquid": res.n_liquid,
    }


def main() -> int:
    import MDAnalysis as mda

    u = mda.Universe(str(GRO))
    n_atoms = len(ic.select_water_oxygens(u))
    box_dimensions = u.dimensions.copy()

    print("=" * 70)
    print("Phase 0 calibration: liquid / Ih / Ic separation gate")
    print("=" * 70)

    print("\n--- (1) Liquid reference (randomized-position negative control) ---")
    liquid = liquid_reference(n_atoms, box_dimensions)
    for r in liquid:
        print(
            f"  seed={r['seed']}: n_max={r['n_max']:4d} chi={r['chi']} "
            f"n_ih={r['n_ih']:4d} n_ic={r['n_ic']:4d} "
            f"interfacial={r['n_interfacial_ice']:4d} liquid={r['n_liquid']:4d}"
        )
    liquid_n_max = np.array([r["n_max"] for r in liquid])

    print("\n--- (2) Bulk Ih reference (real seeded mW crystallization trajectory) ---")
    ih_sweep, nn_distance = ih_reference_and_nn_distance()
    for r in ih_sweep:
        print(
            f"  frame={r['frame']:4d}: n_max={r['n_max']:4d} chi={r['chi']:.4f} "
            f"n_ih={r['n_ih']:4d} n_ic={r['n_ic']:4d} "
            f"interfacial={r['n_interfacial_ice']:4d} liquid={r['n_liquid']:4d}"
        )
    print(f"  empirical O-O NN distance (from grown crystal): {nn_distance:.4f} Angstrom")
    ih_final = ih_sweep[-1]

    print("\n--- (3) Bulk Ic reference (synthetic diamond-cubic lattice, ASE) ---")
    ic_result = ic_reference(nn_distance)
    print(
        f"  lattice_constant={ic_result['lattice_constant']:.4f} A, "
        f"n_atoms={ic_result['n_atoms']}: n_max={ic_result['n_max']} "
        f"chi={ic_result['chi']:.4f} n_ih={ic_result['n_ih']} n_ic={ic_result['n_ic']} "
        f"interfacial={ic_result['n_interfacial_ice']} liquid={ic_result['n_liquid']}"
    )

    print("\n" + "=" * 70)
    print("Go/no-go gate checks")
    print("=" * 70)

    checks = []

    c1 = bool(np.all(liquid_n_max < 20)) and bool(np.all(np.isnan([r["chi"] for r in liquid])))
    checks.append(("Liquid: n_max stays near noise floor (<20) and chi undefined (NaN)", c1))

    c2 = ih_final["n_max"] > 0.9 * n_atoms
    checks.append((f"Ih: n_max grows to >90% of the box ({ih_final['n_max']}/{n_atoms})", c2))

    c3 = all(r["chi"] == 0.0 for r in ih_sweep)
    checks.append(("Ih: chi == 0.0 at every sampled frame (no false-positive cubic labels)", c3))

    c4 = ic_result["chi"] == 1.0
    checks.append(("Ic (synthetic): chi == 1.0 (no false-positive hexagonal labels)", c4))

    c5 = ic_result["n_max"] == ic_result["n_atoms"]
    checks.append(("Ic (synthetic): n_max spans the full synthetic crystal", c5))

    all_pass = True
    for description, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        all_pass = all_pass and passed

    print("\nOVERALL:", "PASS - go" if all_pass else "FAIL - no-go, revisit thresholds")

    out = {
        "liquid_reference": liquid,
        "ih_reference_sweep": ih_sweep,
        "ih_empirical_nn_distance_angstrom": nn_distance,
        "ic_reference_synthetic": ic_result,
        "gate_checks": [{"description": d, "passed": p} for d, p in checks],
        "overall_pass": all_pass,
    }
    out_path = Path(__file__).resolve().parent / "phase0_calibration_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
