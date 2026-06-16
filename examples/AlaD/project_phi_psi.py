"""Alanine dipeptide phi/psi CV extractor for AutoSampler."""

from __future__ import annotations

import numpy as np


def extract_cvs(trajectories, top_file, conf_file):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_dihedrals

    u = mda.Universe(conf_file, trajectories)
    try:
        ace_c = _single_atom(u, "resname ACE and name C")
        ala_n = _single_atom(u, "resname ALA and name N")
        ala_ca = _single_atom(u, "resname ALA and name CA")
        ala_c = _single_atom(u, "resname ALA and name C")
        nme_n = _single_atom(u, "resname NME and name N")

        cvs = np.zeros((u.trajectory.n_frames, 2), dtype=np.float32)
        for frame_index, ts in enumerate(u.trajectory):
            phi = calc_dihedrals(
                ace_c.positions,
                ala_n.positions,
                ala_ca.positions,
                ala_c.positions,
                box=ts.dimensions,
            )[0]
            psi = calc_dihedrals(
                ala_n.positions,
                ala_ca.positions,
                ala_c.positions,
                nme_n.positions,
                box=ts.dimensions,
            )[0]
            cvs[frame_index] = [phi, psi]
        return cvs
    finally:
        u.trajectory.close()


def _single_atom(universe, selection):
    atoms = universe.select_atoms(selection)
    if atoms.n_atoms != 1:
        raise ValueError(f"Selection {selection!r} matched {atoms.n_atoms} atoms.")
    return atoms
