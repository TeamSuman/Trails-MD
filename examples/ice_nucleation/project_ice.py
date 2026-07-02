"""Fixed ice-nucleation CV for Trails-MD: [n_max, chi] per frame.

Implements the Trails-MD ``project_file`` / ``extract_cvs`` plug-in contract
(see examples/AlaD/project_phi_psi.py, examples/AIB9/project_vae_latent.py)
using the physics-based descriptors from ice_descriptors.py:

  n_max -- size of the largest spatially connected cluster of ice-like water
           molecules (nucleation progress).
  chi   -- cubicity, n_Ic / (n_Ic + n_Ih), from CHILL+ classification
           (polymorph selectivity: chi -> 0 selects hexagonal ice, chi -> 1
           selects cubic ice).

Wire this into a Trails-MD input.yaml via:

    system:
      project_file: project_ice.py
    space_mode: fixed
    spawning:
      search_mode: target
      target: [<n_max_target>, <chi_target>]

See docs/ice_nucleation_cv_protocol.md for the full protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ice_descriptors as ic  # noqa: E402


def extract_cvs(trajectories, top_file, conf_file) -> np.ndarray:
    """Return fixed 2D CVs: [n_max, chi] per frame.

    Args:
        trajectories: trajectory file path(s) (Trails-MD passes a list).
        top_file: topology file (unused directly; conf_file carries the
            structure needed by MDAnalysis, matching the other examples).
        conf_file: structure/topology file MDAnalysis can pair with
            ``trajectories`` to build a Universe.
    """
    import MDAnalysis as mda

    u = mda.Universe(conf_file, trajectories)
    try:
        ow = ic.select_water_oxygens(u)
        cvs = np.zeros((u.trajectory.n_frames, 2), dtype=np.float32)
        for frame_index, ts in enumerate(u.trajectory):
            result = ic.ice_cv(ow.positions, ts.dimensions)
            # chi is NaN when the frame has no hex/cubic molecules at all
            # (pure liquid, before any nucleus has formed). Substitute 0.0 so
            # the CV array stays finite for Trails-MD's distance-to-target
            # and binning computations; n_max already correctly reads ~0 in
            # that regime, so this does not create a false polymorph signal.
            chi = result.chi if np.isfinite(result.chi) else 0.0
            cvs[frame_index] = [result.n_max, chi]
        return cvs
    finally:
        u.trajectory.close()
