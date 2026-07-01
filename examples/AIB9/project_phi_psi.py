"""AIB9 phi/psi projector for fixed-space Trails-MD runs.

The paper tracks 18 dihedrals ordered as phi1, psi1, ..., phi9, psi9 and
illustrates the free-energy surface using residue 5.  Trails-MD's fixed density
spawner is most practical in 2D, so extract_cvs returns the residue-5 phi/psi
pair by default.  Use extract_all_phi_psi() when the full 18D paper coordinate
set is needed for analysis.
"""

from __future__ import annotations

import os

import numpy as np


DEFAULT_RESIDUE_INDEX = 5


def _single_atom(residue, name: str):
    atoms = residue.atoms.select_atoms(f"name {name}")
    if atoms.n_atoms != 1:
        raise ValueError(
            f"Residue {residue.resid} {residue.resname} atom {name!r} "
            f"matched {atoms.n_atoms} atoms."
        )
    return atoms


def _aib_residues(universe):
    residues = list(universe.select_atoms("resname AIB").residues)
    if len(residues) != 9:
        raise ValueError(f"Expected 9 AIB residues, found {len(residues)}.")
    return residues


def _dihedral_groups(residues):
    phi_groups = []
    psi_groups = []
    for index, residue in enumerate(residues):
        previous_c = _single_atom(residue, "CY") if index == 0 else _single_atom(residues[index - 1], "C")
        next_n = _single_atom(residue, "NT") if index == len(residues) - 1 else _single_atom(residues[index + 1], "N")
        n = _single_atom(residue, "N")
        ca = _single_atom(residue, "CA")
        c = _single_atom(residue, "C")
        phi_groups.append((previous_c, n, ca, c))
        psi_groups.append((n, ca, c, next_n))
    return phi_groups, psi_groups


def extract_all_phi_psi(trajectories: list[str], top_file: str, conf_file: str) -> np.ndarray:
    """Return all AIB9 phi/psi dihedrals in radians.

    Output columns are phi1, psi1, phi2, psi2, ..., phi9, psi9.
    """

    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_dihedrals

    universe = mda.Universe(conf_file, trajectories)
    residues = _aib_residues(universe)
    phi_groups, psi_groups = _dihedral_groups(residues)

    cvs = np.zeros((universe.trajectory.n_frames, 18), dtype=np.float32)
    try:
        for frame_index, ts in enumerate(universe.trajectory):
            row = []
            for phi_group, psi_group in zip(phi_groups, psi_groups):
                phi = calc_dihedrals(
                    phi_group[0].positions,
                    phi_group[1].positions,
                    phi_group[2].positions,
                    phi_group[3].positions,
                    box=ts.dimensions,
                )[0]
                psi = calc_dihedrals(
                    psi_group[0].positions,
                    psi_group[1].positions,
                    psi_group[2].positions,
                    psi_group[3].positions,
                    box=ts.dimensions,
                )[0]
                row.extend([phi, psi])
            cvs[frame_index] = row
    finally:
        universe.trajectory.close()

    return cvs


def extract_cvs(trajectories: list[str], top_file: str, conf_file: str) -> np.ndarray:
    """Return one physical 2D phi/psi pair in radians for fixed-space sampling."""

    residue_index = int(os.environ.get("AIB9_PHI_PSI_RESIDUE", DEFAULT_RESIDUE_INDEX))
    if residue_index < 1 or residue_index > 9:
        raise ValueError("AIB9_PHI_PSI_RESIDUE must be between 1 and 9.")

    all_angles = extract_all_phi_psi(
        trajectories=trajectories,
        top_file=top_file,
        conf_file=conf_file,
    )
    start = 2 * (residue_index - 1)
    return all_angles[:, start : start + 2]
