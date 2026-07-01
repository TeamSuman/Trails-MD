
import MDAnalysis as mda  # type: ignore
import numpy as np
from MDAnalysis.lib.distances import distance_array  # type: ignore


def _load_universe(topology: str, trajectories: list[str] | str, **kwargs) -> mda.Universe:
    """Helper to load an MDAnalysis Universe, automatically passing format="TRJ" for Amber ASCII trajectories."""
    first_traj = None
    if isinstance(trajectories, list):
        if len(trajectories) > 0:
            first_traj = trajectories[0]
    else:
        first_traj = trajectories

    if first_traj and str(first_traj).endswith((".crd", ".mdcrd", ".trj")):
        kwargs.setdefault("format", "TRJ")
    return mda.Universe(topology, trajectories, **kwargs)


class FeatureExtractor:
    """Extracts features from MD trajectories for dimensionality reduction."""

    def __init__(self, topology: str, selection: str | None = None):
        self.topology = topology
        self.selection = selection if selection else "protein and not (type H)"

    def extract_pairwise_distances(self, trajectories: list[str]) -> np.ndarray:
        """Calculate pairwise distances for a list of trajectory files."""
        # Check if the topology and trajectories exist
        try:
            u = _load_universe(self.topology, trajectories)
        except Exception as e:
            raise ValueError(f"Failed to load universe with top={self.topology}, trajs={trajectories}: {e}") from e

        ag = u.select_atoms(self.selection)
        num_pairs = ag.n_atoms * (ag.n_atoms - 1) // 2

        try:
            dist_list = np.zeros((u.trajectory.n_frames, num_pairs), dtype=np.float32)
            for j, _ts in enumerate(u.trajectory):
                r = distance_array(ag, ag, box=u.dimensions, backend="OpenMP")
                r = r[np.triu_indices(r.shape[0], k=1)]
                dist_list[j] = r
        finally:
            u.trajectory.close()

        return dist_list

    def extract_fitted_coords(self, trajectories: list[str]) -> np.ndarray:
        """Calculate flattened Cartesian coordinates after RMSD fitting to the reference topology."""
        from MDAnalysis.analysis import align

        try:
            # ref universe for the single frame topology
            ref = mda.Universe(self.topology)
            u = _load_universe(self.topology, trajectories)
        except Exception as e:
            raise ValueError(f"Failed to load universe for fitted coords extraction: {e}") from e

        ag = u.select_atoms(self.selection)
        num_features = ag.n_atoms * 3

        try:
            coord_list = np.zeros((u.trajectory.n_frames, num_features), dtype=np.float32)
            for j, _ts in enumerate(u.trajectory):
                # Align the current frame in memory to the reference structure
                align.alignto(u, ref, select=self.selection)
                # Store the flattened coordinates of the selection
                coord_list[j] = ag.positions.flatten()
        finally:
            u.trajectory.close()

        return coord_list

    def extract_aib9_phi_psi(self, trajectories: list[str]) -> np.ndarray:
        """Extract the 18 AIB9 phi/psi dihedral angles used in the DDPM paper.

        Returns radians ordered as:
        phi1, psi1, phi2, psi2, ..., phi9, psi9.
        The capped N/C terminal atoms are used for phi1 and psi9.
        """
        from MDAnalysis.lib.distances import calc_dihedrals

        try:
            u = _load_universe(self.topology, trajectories)
        except Exception as e:
            raise ValueError(f"Failed to load universe for AIB9 phi/psi extraction: {e}") from e

        residues = list(u.select_atoms("resname AIB").residues)
        if len(residues) != 9:
            raise ValueError(f"AIB9 phi/psi extraction expected 9 AIB residues, found {len(residues)}.")

        def atom(residue, name: str):
            atoms = residue.atoms.select_atoms(f"name {name}")
            if atoms.n_atoms != 1:
                raise ValueError(
                    f"Residue {residue.resid} {residue.resname} atom {name!r} "
                    f"matched {atoms.n_atoms} atoms."
                )
            return atoms

        phi_atoms = []
        psi_atoms = []
        for index, residue in enumerate(residues):
            previous_c = atom(residue, "CY") if index == 0 else atom(residues[index - 1], "C")
            next_n = atom(residue, "NT") if index == len(residues) - 1 else atom(residues[index + 1], "N")
            n = atom(residue, "N")
            ca = atom(residue, "CA")
            c = atom(residue, "C")
            phi_atoms.append((previous_c, n, ca, c))
            psi_atoms.append((n, ca, c, next_n))

        try:
            values = np.zeros((u.trajectory.n_frames, 18), dtype=np.float32)
            for frame_index, ts in enumerate(u.trajectory):
                row = []
                for phi_group, psi_group in zip(phi_atoms, psi_atoms, strict=False):
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
                values[frame_index] = row
        finally:
            u.trajectory.close()

        return values

    def extract_rg_rmsd(self, trajectories: list[str], reference_pdb: str) -> np.ndarray:
        """Calculate Radius of Gyration and RMSD (to reference) for physical fixed-space projection."""
        from MDAnalysis.analysis import rms  # type: ignore

        try:
            u = _load_universe(self.topology, trajectories)
            ref = _load_universe(self.topology, reference_pdb)
        except Exception as e:
            raise ValueError(f"Failed to load universe for Rg/RMSD: {e}") from e

        protein = u.select_atoms("protein")

        try:
            # Calculate Rg
            rg_values = np.zeros(u.trajectory.n_frames, dtype=np.float32)
            for j, _ts in enumerate(u.trajectory):
                rg_values[j] = protein.radius_of_gyration()

            # Calculate RMSD
            # Default to CA backbone for stable RMSD
            R = rms.RMSD(u, ref, select="protein", groupselections=["name CA"])
            R.run()
            # R.results.rmsd has shape (n_frames, 3+) where index 2 is the selection RMSD
            rmsd_values = R.results.rmsd[:, 2]
        finally:
            u.trajectory.close()
            ref.trajectory.close()

        # Combine into (n_frames, 2) shape: [Rg, RMSD]
        return np.column_stack((rg_values, rmsd_values))

    def extract_positions_by_indices(self, trajectories: list[str], indices: list[int]) -> list:
        """Return full-system OpenMM positions for selected global frame indices."""
        from MDAnalysis.coordinates.XTC import XTCReader  # type: ignore
        from openmm.unit import nanometer  # type: ignore

        # Map global indices to their original order to return them correctly
        sorted_requests = sorted(list(enumerate(indices)), key=lambda x: x[1])
        states_dict = {}

        current_global_frame = 0
        traj_idx = 0
        u = None

        try:
            for original_order, target_index in sorted_requests:
                if target_index < 0:
                    raise IndexError(f"Spawn frame index {target_index} cannot be negative.")

                while True:
                    if traj_idx >= len(trajectories):
                        raise IndexError(f"Spawn frame index {target_index} is outside trajectory range.")

                    # Compute length extremely fast without parsing topology
                    if str(trajectories[traj_idx]).endswith(".xtc"):
                        with XTCReader(trajectories[traj_idx]) as reader:
                            traj_length = reader.n_frames
                    else:
                        u_temp = _load_universe(self.topology, trajectories[traj_idx])
                        traj_length = u_temp.trajectory.n_frames
                        u_temp.trajectory.close()

                    if target_index < current_global_frame + traj_length:
                        # Load topology ONLY for the target trajectory
                        if u is None or getattr(u, "_current_traj_idx", -1) != traj_idx:
                            is_crd = str(trajectories[traj_idx]).endswith((".crd", ".mdcrd", ".trj"))
                            if u is None:
                                u = _load_universe(self.topology, trajectories[traj_idx])
                            else:
                                u.load_new(trajectories[traj_idx], format="TRJ" if is_crd else None)
                            u._current_traj_idx = traj_idx

                        local_index = target_index - current_global_frame
                        ts = u.trajectory[local_index]
                        positions_nm = u.atoms.positions.copy() * 0.1
                        states_dict[original_order] = {
                            "positions": positions_nm * nanometer,
                            "box_vectors": _box_vectors_from_dimensions(ts.dimensions),
                        }
                        break
                    else:
                        current_global_frame += traj_length
                        traj_idx += 1

        finally:
            if u is not None:
                u.trajectory.close()

        # Reconstruct the results in the original order requested
        return [states_dict[i] for i in range(len(indices))]


def _box_vectors_from_dimensions(dimensions):
    """Convert MDAnalysis unit-cell dimensions to OpenMM box vectors."""
    from math import cos, radians, sin, sqrt

    from openmm import Vec3  # type: ignore
    from openmm.unit import nanometer  # type: ignore

    if dimensions is None or len(dimensions) < 6 or np.any(np.asarray(dimensions[:3]) <= 0):
        return None

    lx, ly, lz, alpha, beta, gamma = (float(value) for value in dimensions[:6])
    lx *= 0.1
    ly *= 0.1
    lz *= 0.1
    if all(abs(angle - 90.0) < 1e-3 for angle in (alpha, beta, gamma)):
        return (
            Vec3(lx, 0.0, 0.0) * nanometer,
            Vec3(0.0, ly, 0.0) * nanometer,
            Vec3(0.0, 0.0, lz) * nanometer,
        )

    alpha = radians(alpha)
    beta = radians(beta)
    gamma = radians(gamma)

    ax, ay, az = lx, 0.0, 0.0
    bx, by, bz = ly * cos(gamma), ly * sin(gamma), 0.0
    cx = lz * cos(beta)
    cy = lz * (cos(alpha) - cos(beta) * cos(gamma)) / max(sin(gamma), 1e-12)
    cz = sqrt(max(lz * lz - cx * cx - cy * cy, 0.0))
    return (
        Vec3(ax, ay, az) * nanometer,
        Vec3(bx, by, bz) * nanometer,
        Vec3(cx, cy, cz) * nanometer,
    )
