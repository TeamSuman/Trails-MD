"""GROMACS MD engine for AutoSampler.

Runs production MD using the GROMACS ``gmx`` binary (``grompp`` + ``mdrun``)
via subprocess.  The engine writes GROMACS XTC trajectories natively — no
post-run format conversion is needed.

Start coordinates accepted by ``run_production`` can be:

* A file path (str or Path) pointing to a GROMACS GRO / PDB file.
* A dict ``{"positions": <OpenMM Quantity>, "box_vectors": ...}`` as returned
  by ``FeatureExtractor.extract_positions_by_indices``.  The engine converts
  the OpenMM-unit positions to Angstroms, loads atom metadata from the
  original GRO template via MDAnalysis, and writes a fresh GRO file.

The two-step GROMACS pipeline is:

1. ``gmx grompp``  — preprocesses MDP + GRO + TOP into a run-input TPR.
2. ``gmx mdrun``   — executes the production run and writes XTC.

GPU device isolation uses the ``CUDA_VISIBLE_DEVICES`` environment variable,
which GROMACS respects automatically.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .base import MDEngine


class GromacsEngine(MDEngine):
    """GROMACS pmemd-based implementation of MDEngine strategy.

    Parameters
    ----------
    temperature:
        Target temperature in Kelvin (default 300 K).
    pressure:
        Reference pressure in bar (default 1.0 bar).
    dt:
        Integration timestep in picoseconds (default 0.002 ps).
    npt:
        Run under constant pressure (NPT) using Parrinello-Rahman if True,
        else NVT.
    gromacs_executable:
        Name or full path of the GROMACS driver binary
        (default ``"gmx"``).  Use ``"gmx_mpi"`` for MPI builds.
    gromacs_include_dir:
        Optional path added to ``GMXLIB`` so that ``grompp`` can find custom
        force-field files.  Standard GROMACS force fields are found
        automatically via the binary installation.
    """

    def __init__(
        self,
        temperature: float = 300.0,
        pressure: float = 1.0,
        dt: float = 0.002,
        npt: bool = False,
        gromacs_executable: str = "gmx",
        gromacs_include_dir: Optional[str] = None,
        gromacs_mdrun_nb: Optional[str] = None,
        gromacs_mdrun_pme: Optional[str] = None,
        gromacs_mdrun_update: Optional[str] = None,
        gromacs_mdrun_bonded: Optional[str] = None,
        gromacs_mdrun_pin: Optional[str] = None,
        gromacs_mdrun_ntmpi: int = 1,
        gromacs_mdrun_ntomp: Optional[int] = None,
        gromacs_mdrun_extra_args: Optional[list[str]] = None,
        **kwargs,  # absorb OpenMM / Amber specific kwargs
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt
        self.npt = npt
        self.gromacs_executable = gromacs_executable
        self.gromacs_include_dir = gromacs_include_dir
        self.gromacs_mdrun_nb = gromacs_mdrun_nb
        self.gromacs_mdrun_pme = gromacs_mdrun_pme
        self.gromacs_mdrun_update = gromacs_mdrun_update
        self.gromacs_mdrun_bonded = gromacs_mdrun_bonded
        self.gromacs_mdrun_pin = gromacs_mdrun_pin
        self.gromacs_mdrun_ntmpi = gromacs_mdrun_ntmpi
        self.gromacs_mdrun_ntomp = gromacs_mdrun_ntomp
        self.gromacs_mdrun_extra_args = list(gromacs_mdrun_extra_args or [])

        # Set after prepare()
        self.topology_file: Optional[Path] = None
        self.start_coords_file: Optional[Path] = None
        self.mdp_template: Optional[Path] = None
        self.positions: Optional[str] = None  # str path for first-iteration walkers

    # ------------------------------------------------------------------
    # MDEngine interface
    # ------------------------------------------------------------------

    def prepare(
        self, conf: Path, top: Path, system_file: Optional[Path] = None
    ) -> None:
        """Validate GROMACS input files and store run-time parameters.

        Parameters
        ----------
        conf:
            Path to the starting GRO / PDB coordinate file.
        top:
            Path to the GROMACS topology (``.top``) file.
        system_file:
            Optional path to a custom MDP template file.  Placeholders
            ``{steps}``, ``{dt}``, ``{temp}``, ``{stride}``, ``{pressure}``,
            ``{pcoupl}``, ``{gen_vel}`` are substituted at run time.
            If omitted a built-in default MDP is generated.
        """
        self.topology_file = Path(top)
        self.start_coords_file = Path(conf)
        self.mdp_template = Path(system_file) if system_file is not None else None

        if not self.topology_file.exists():
            raise FileNotFoundError(
                f"GROMACS topology file not found: {self.topology_file}"
            )
        if not self.start_coords_file.exists():
            raise FileNotFoundError(
                f"GROMACS coordinate file not found: {self.start_coords_file}"
            )
        if self.mdp_template is not None and not self.mdp_template.exists():
            raise FileNotFoundError(f"MDP template file not found: {self.mdp_template}")

        # Expose the initial coordinate path so the runner can seed walkers
        # (mirrors OpenMMEngine.positions for the first iteration).
        self.positions = str(self.start_coords_file)
        logging.info(
            "GromacsEngine prepared: top=%s  conf=%s  mdp=%s",
            self.topology_file,
            self.start_coords_file,
            self.mdp_template,
        )

    def run_production(
        self,
        run_index: int,
        start_coords,
        steps: int,
        traj_out: Path,
        stride: int,
        device_index: int,
    ) -> bool:
        """Execute one grompp + mdrun cycle and write an XTC trajectory.

        Parameters
        ----------
        run_index:
            Walker index used to name per-run scratch files.
        start_coords:
            Either a file path (str / Path) to a GRO file, or a dict
            ``{"positions": ..., "box_vectors": ...}`` with OpenMM-unit data.
        steps:
            Number of MD integration steps.
        traj_out:
            Destination XTC file path (framework convention).
        stride:
            Write a frame every *stride* steps (``nstxout-compressed``).
        device_index:
            CUDA device index, mapped to ``CUDA_VISIBLE_DEVICES``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if either grompp or mdrun failed.
        """
        traj_out = Path(traj_out)
        workdir = traj_out.parent
        workdir.mkdir(parents=True, exist_ok=True)

        # Remove stale output
        if traj_out.exists():
            traj_out.unlink()

        # ── Prepare GRO start file ────────────────────────────────────────
        gro_start = workdir / f"_start_{run_index}.gro"
        positions, box_vectors = self._split_start_state(start_coords)

        if positions is None:
            # No positions provided: fall back to the prepared coordinates
            shutil.copy(str(self.start_coords_file), str(gro_start))
        elif isinstance(positions, (str, Path)) and Path(str(positions)).exists():
            shutil.copy(str(positions), str(gro_start))
        else:
            # OpenMM-format positions (Quantity in nm) → Angstroms → GRO
            pos_ang = self._openmm_positions_to_angstrom(positions)
            box_ang = self._openmm_box_to_angstrom(box_vectors)
            self._write_gro(str(gro_start), pos_ang, box_ang)

        # ── Write MDP file ────────────────────────────────────────────────
        mdp_file = workdir / f"_md_{run_index}.mdp"
        self._write_mdp(str(mdp_file), steps, stride)

        # ── Scratch output paths ──────────────────────────────────────────
        tpr_file = workdir / f"_run_{run_index}.tpr"
        xtc_tmp = workdir / f"_traj_{run_index}.xtc"
        final_gro = workdir / f"_final_{run_index}.gro"
        edr_file = workdir / f"_run_{run_index}.edr"
        log_file = workdir / f"_run_{run_index}.log"

        # ── Build subprocess environment ──────────────────────────────────
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        if self.gromacs_include_dir:
            existing = env.get("GMXLIB", "")
            env["GMXLIB"] = (
                f"{self.gromacs_include_dir}:{existing}".rstrip(":")
                if existing
                else self.gromacs_include_dir
            )

        # ── Step 1: grompp ────────────────────────────────────────────────
        grompp_cmd = [
            self.gromacs_executable,
            "grompp",
            "-f",
            str(mdp_file),
            "-c",
            str(gro_start),
            "-p",
            str(self.topology_file),
            "-o",
            str(tpr_file),
            "-maxwarn",
            "5",
        ]

        try:
            subprocess.run(
                grompp_cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logging.error(
                "GromacsEngine grompp run %d failed (exit %d):\n%s",
                run_index,
                exc.returncode,
                exc.stderr,
            )
            return False
        except FileNotFoundError:
            logging.error(
                "GromacsEngine: executable %r not found on PATH.",
                self.gromacs_executable,
            )
            return False

        # ── Step 2: mdrun ─────────────────────────────────────────────────
        mdrun_cmd = [
            self.gromacs_executable,
            "mdrun",
            "-s",
            str(tpr_file),
            "-x",
            str(xtc_tmp),
            "-c",
            str(final_gro),
            "-e",
            str(edr_file),
            "-g",
            str(log_file),
        ]
        mdrun_cmd.extend(self._mdrun_option_args())

        try:
            subprocess.run(
                mdrun_cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logging.error(
                "GromacsEngine mdrun run %d failed (exit %d):\n%s",
                run_index,
                exc.returncode,
                exc.stderr,
            )
            return False
        except FileNotFoundError:
            logging.error(
                "GromacsEngine: executable %r not found on PATH.",
                self.gromacs_executable,
            )
            return False

        # ── Move XTC to canonical output path ─────────────────────────────
        shutil.move(str(xtc_tmp), str(traj_out))

        # ── Clean up scratch files ────────────────────────────────────────
        for path in (gro_start, mdp_file, tpr_file, final_gro, edr_file, log_file):
            if path.exists():
                path.unlink()

        return True

    def _mdrun_option_args(self) -> list[str]:
        args = ["-ntmpi", str(self.gromacs_mdrun_ntmpi)]
        if self.gromacs_mdrun_ntomp is not None:
            args.extend(["-ntomp", str(self.gromacs_mdrun_ntomp)])
        for flag, value in (
            ("-nb", self.gromacs_mdrun_nb),
            ("-pme", self.gromacs_mdrun_pme),
            ("-update", self.gromacs_mdrun_update),
            ("-bonded", self.gromacs_mdrun_bonded),
            ("-pin", self.gromacs_mdrun_pin),
        ):
            if value:
                args.extend([flag, value])
        args.extend(str(value) for value in self.gromacs_mdrun_extra_args)
        return args

    # ------------------------------------------------------------------
    # Input / output helpers
    # ------------------------------------------------------------------

    def _write_mdp(self, filepath: str, steps: int, stride: int) -> None:
        """Write a GROMACS MDP run-parameters file.

        If ``mdp_template`` was set in ``prepare()``, the file is read as a
        Python format-string template with the following named replacements:

        ``{steps}``, ``{dt}``, ``{temp}``, ``{stride}``, ``{pressure}``,
        ``{pcoupl}``, ``{tau_p}``, ``{gen_vel}``

        If no template is provided, a built-in default is generated that is
        compatible with spawned walkers (coordinates only, no velocities):

        * NVT   — V-rescale thermostat, ``gen_vel = yes``
        * NPT   — adds Parrinello-Rahman barostat
        """
        pcoupl = "Parrinello-Rahman" if self.npt else "no"
        tau_p = "2.0" if self.npt else "0.0"
        pres_line = (
            (
                f"pcoupltype              = isotropic\n"
                f"tau_p                   = {tau_p}\n"
                f"ref_p                   = {self.pressure:.4f}\n"
                f"compressibility         = 4.5e-5\n"
            )
            if self.npt
            else ""
        )

        if self.mdp_template is not None:
            template_path = Path(self.mdp_template)
            if not template_path.exists():
                raise FileNotFoundError(f"MDP template not found: {self.mdp_template}")
            template = template_path.read_text()
            content = template.format(
                steps=steps,
                dt=self.dt,
                temp=self.temperature,
                stride=stride,
                pressure=self.pressure,
                pcoupl=pcoupl,
                tau_p=tau_p,
                gen_vel="yes",
            )
        else:
            content = (
                f"title                   = AutoSampler MD run\n"
                f"integrator              = md\n"
                f"nsteps                  = {steps}\n"
                f"dt                      = {self.dt:.6f}\n"
                f"\n"
                f"; Output\n"
                f"nstxout                 = 0\n"
                f"nstvout                 = 0\n"
                f"nstfout                 = 0\n"
                f"nstenergy               = {steps}\n"
                f"nstlog                  = {steps}\n"
                f"nstxout-compressed      = {stride}\n"
                f"compressed-x-grps       = System\n"
                f"\n"
                f"; Bond parameters\n"
                f"continuation            = no\n"
                f"constraint_algorithm    = lincs\n"
                f"constraints             = h-bonds\n"
                f"lincs_iter              = 1\n"
                f"lincs_order             = 4\n"
                f"\n"
                f"; Neighbour searching\n"
                f"cutoff-scheme           = Verlet\n"
                f"nstlist                 = 10\n"
                f"rcoulomb                = 1.0\n"
                f"rvdw                    = 1.0\n"
                f"\n"
                f"; Electrostatics\n"
                f"coulombtype             = PME\n"
                f"pme_order               = 4\n"
                f"fourierspacing          = 0.16\n"
                f"\n"
                f"; Temperature coupling\n"
                f"tcoupl                  = V-rescale\n"
                f"tc-grps                 = System\n"
                f"tau_t                   = 0.1\n"
                f"ref_t                   = {self.temperature:.2f}\n"
                f"\n"
                f"; Pressure coupling\n"
                f"pcoupl                  = {pcoupl}\n"
                f"{pres_line}"
                f"\n"
                f"; Periodic boundary conditions\n"
                f"pbc                     = xyz\n"
                f"DispCorr                = EnerPres\n"
                f"\n"
                f"; Velocity generation — always regenerate for spawned frames\n"
                f"gen_vel                 = yes\n"
                f"gen_temp                = {self.temperature:.2f}\n"
                f"gen_seed                = -1\n"
            )

        Path(filepath).write_text(content)

    def _write_gro(
        self,
        filepath: str,
        positions_ang: np.ndarray,
        box_ang: Optional[np.ndarray] = None,
    ) -> None:
        """Write a GROMACS GRO file with updated positions.

        Atom and residue metadata are preserved from the original coordinate
        file set by ``prepare()``.  The new positions (in Angstroms) replace
        the stored coordinates.  If *box_ang* is provided the box vectors are
        updated accordingly.

        Parameters
        ----------
        filepath:
            Destination ``.gro`` file path.
        positions_ang:
            Array of shape ``(natoms, 3)`` in **Angstroms** (MDAnalysis native
            units).
        box_ang:
            Array ``[a, b, c, alpha, beta, gamma]`` in Angstroms / degrees, or
            ``None`` to keep the box from the template.
        """
        import MDAnalysis as mda  # type: ignore

        u = mda.Universe(str(self.start_coords_file))
        try:
            u.atoms.positions = positions_ang
            if box_ang is not None:
                u.trajectory.ts.dimensions = np.asarray(box_ang[:6], dtype=np.float32)
            with mda.Writer(filepath, n_atoms=u.atoms.n_atoms) as writer:
                writer.write(u)
        finally:
            u.trajectory.close()

    # ------------------------------------------------------------------
    # Static helpers — position-format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _split_start_state(start_coords):
        """Split *start_coords* into (positions, box_vectors).

        Mirrors the helper in ``OpenMMEngine`` / ``AmberEngine`` so all three
        engines share the same walker-dict interface.
        """
        if isinstance(start_coords, dict):
            return start_coords.get("positions"), start_coords.get("box_vectors")
        return start_coords, None

    @staticmethod
    def _openmm_positions_to_angstrom(positions) -> np.ndarray:
        """Convert OpenMM Quantity positions (nm) to a numpy array in Angstroms.

        Falls back gracefully when OpenMM is not importable, treating the
        input as a raw array already in nm.
        """
        try:
            from openmm.unit import is_quantity, nanometer  # type: ignore

            if is_quantity(positions):
                pos_nm = positions.value_in_unit(nanometer)
                return (
                    np.array([[v[0], v[1], v[2]] for v in pos_nm], dtype=np.float64)
                    * 10.0
                )
        except ImportError:
            pass
        return (
            np.array([[v[0], v[1], v[2]] for v in positions], dtype=np.float64) * 10.0
        )

    @staticmethod
    def _openmm_box_to_angstrom(box_vectors) -> Optional[np.ndarray]:
        """Convert an OpenMM box-vectors tuple to ``[a, b, c, 90, 90, 90]`` in Angstroms.

        Only orthogonal boxes are currently supported.
        """
        if box_vectors is None:
            return None
        try:
            from openmm.unit import is_quantity, nanometer  # type: ignore

            bv = []
            for vec in box_vectors:
                if is_quantity(vec):
                    vec = vec.value_in_unit(nanometer)
                bv.append([float(vec[0]), float(vec[1]), float(vec[2])])
            bv_ang = np.array(bv, dtype=np.float64) * 10.0  # nm → Å
            a, b, c = bv_ang[0][0], bv_ang[1][1], bv_ang[2][2]
            return np.array([a, b, c, 90.0, 90.0, 90.0], dtype=np.float64)
        except (ImportError, TypeError, IndexError) as exc:
            logging.warning("GromacsEngine: could not parse box vectors (%s).", exc)
            return None
