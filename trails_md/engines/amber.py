"""Amber MD engine for Trails-MD.

Runs production MD using pmemd (CUDA or CPU) via subprocess.  Trajectory
output is written in Amber NetCDF (.nc) format and then converted to XTC so
the rest of the framework – which expects .xtc files – remains unchanged.

Start coordinates accepted by ``run_production`` can be:

* A file path (str or Path) pointing to an Amber RST7 / INPCRD file.
* A dict ``{"positions": <OpenMM Quantity>, "box_vectors": ...}`` as returned
  by ``FeatureExtractor.extract_positions_by_indices``.  The engine converts
  the OpenMM-unit positions to Angstroms and writes a fresh RST7 file.

"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import MDEngine, box_vectors_to_abc_angles, md_subprocess_timeout


def resolve_amber_trajectory_format(
    amber_trajectory_format: str,
    amber_executable: str,
) -> str:
    """Resolve Amber trajectory output format for the selected executable."""
    amber_trajectory_format = amber_trajectory_format.lower()
    if amber_trajectory_format == "auto":
        executable_name = Path(amber_executable).name.lower()
        return "netcdf" if "cuda" in executable_name else "ascii"
    if amber_trajectory_format not in {"netcdf", "ascii"}:
        raise ValueError(
            "amber_trajectory_format must be 'auto', 'netcdf', or 'ascii'"
        )
    return amber_trajectory_format


def amber_trajectory_suffix(
    amber_trajectory_format: str,
    amber_executable: str,
) -> str:
    """Return Trails-MD's canonical Amber trajectory suffix."""
    if (
        resolve_amber_trajectory_format(amber_trajectory_format, amber_executable)
        == "ascii"
    ):
        return "mdcrd"
    return "nc"


@lru_cache(maxsize=1)
def _find_libnvjitlink_dir() -> str | None:
    """Find the directory containing libnvJitLink.so.12 dynamically."""
    import sys

    # 1. Try to import nvidia.nvjitlink
    try:
        import nvidia.nvjitlink
        nvjitlink_path = Path(nvidia.nvjitlink.__file__).parent / "lib"
        if (nvjitlink_path / "libnvJitLink.so.12").exists():
            return str(nvjitlink_path)
    except ImportError:
        pass

    # 2. Search miniconda envs and pkgs directories
    prefix_path = Path(sys.prefix)
    conda_root = None
    for parent in [prefix_path] + list(prefix_path.parents):
        if (parent / "envs").exists() or (parent / "pkgs").exists():
            conda_root = parent
            break

    if conda_root:
        # Search pkgs first
        for libdir in conda_root.glob("pkgs/libnvjitlink-*/lib"):
            if (libdir / "libnvJitLink.so.12").exists():
                return str(libdir)
        for libdir in conda_root.glob("pkgs/libnvjitlink-*/targets/x86_64-linux/lib"):
            if (libdir / "libnvJitLink.so.12").exists():
                return str(libdir)
        # Then search other envs
        for libdir in conda_root.glob("envs/*/lib"):
            if (libdir / "libnvJitLink.so.12").exists():
                return str(libdir)
        for libdir in conda_root.glob("envs/*/targets/x86_64-linux/lib"):
            if (libdir / "libnvJitLink.so.12").exists():
                return str(libdir)

    # 3. Check some other system paths
    for path in [
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda/lib64",
        "/opt/cuda/lib64",
    ]:
        p = Path(path)
        if (p / "libnvJitLink.so.12").exists():
            return str(p)

    return None


class AmberEngine(MDEngine):
    """Amber pmemd-based implementation of MDEngine strategy.

    Parameters
    ----------
    temperature:
        Target temperature in Kelvin (default 300 K).
    pressure:
        Reference pressure in atmospheres (default 1.0 atm).
    dt:
        Integration timestep in picoseconds (default 0.002 ps).
    npt:
        Run under constant pressure (NPT) if True, else NVT.
    amber_executable:
        Name or full path of the Amber MD executable
        (default ``"pmemd"``).  Use ``"pmemd.cuda"`` for CUDA-enabled runs
        when that executable is installed.
    amber_input_file:
        Optional path to a custom Amber ``.in`` template.  Placeholders
        ``{steps}``, ``{dt}``, ``{temp}``, ``{stride}``, and ``{ntp}`` are
        replaced with run-specific values via ``str.format``.
    """

    def __init__(
        self,
        temperature: float = 300.0,
        pressure: float = 1.0,
        dt: float = 0.002,
        npt: bool = False,
        amber_executable: str = "pmemd",
        amber_input_file: str | None = None,
        amber_extra_args: list[str] | None = None,
        amber_trajectory_format: str = "auto",
        **kwargs,  # absorb OpenMM-specific kwargs (precision, platform_name, …)
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt
        self.npt = npt
        self.amber_executable = amber_executable
        self.amber_input_file = amber_input_file
        self.amber_extra_args = list(amber_extra_args or [])
        self.amber_trajectory_format = amber_trajectory_format.lower()
        self._resolved_trajectory_format()

        # Set after prepare()
        self.topology_file: Path | None = None
        self.start_coords_file: Path | None = None
        self.positions: str | None = None  # str path for first-iteration walkers

    # ------------------------------------------------------------------
    # MDEngine interface
    # ------------------------------------------------------------------

    def prepare(
        self, conf: Path, top: Path, system_file: Path | None = None
    ) -> None:
        """Validate the Amber topology and coordinate files.

        Parameters
        ----------
        conf:
            Path to the starting coordinates (.rst7 / .inpcrd / .pdb).
        top:
            Path to the Amber topology (.prmtop / .parm7).
        system_file:
            Unused for the Amber engine; accepted for interface compatibility.
        """
        self.topology_file = Path(top)
        self.start_coords_file = Path(conf)

        if not self.topology_file.exists():
            raise FileNotFoundError(
                f"Amber topology file not found: {self.topology_file}"
            )
        if not self.start_coords_file.exists():
            raise FileNotFoundError(
                f"Amber coordinate file not found: {self.start_coords_file}"
            )

        # Expose the initial coordinate path so the runner can seed walkers
        # (mirrors OpenMMEngine.positions for the first iteration).
        self.positions = str(self.start_coords_file)
        logging.info(
            "AmberEngine prepared: top=%s  conf=%s",
            self.topology_file,
            self.start_coords_file,
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
        """Execute one production run and write an XTC trajectory.

        Parameters
        ----------
        run_index:
            Walker index used to name per-run scratch files.
        start_coords:
            Either a file path (str / Path) to an RST7 file, or a dict
            ``{"positions": ..., "box_vectors": ...}`` with OpenMM-unit data.
        steps:
            Number of MD integration steps.
        traj_out:
            Destination XTC file path (framework convention).
        stride:
            Write a frame every *stride* steps.
        device_index:
            CUDA device index, mapped to ``CUDA_VISIBLE_DEVICES``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the MD executable failed.
        """
        traj_out = Path(traj_out)
        workdir = traj_out.parent
        workdir.mkdir(parents=True, exist_ok=True)

        # Remove stale output
        if traj_out.exists():
            traj_out.unlink()

        # ── Prepare RST7 start file ───────────────────────────────────────
        rst7_start = workdir / f"_start_{run_index}.rst7"
        positions, box_vectors = self._split_start_state(start_coords)

        if positions is None:
            # No positions provided: fall back to the prepared coordinates
            shutil.copy(str(self.start_coords_file), str(rst7_start))
        elif isinstance(positions, (str, Path)) and Path(str(positions)).exists():
            shutil.copy(str(positions), str(rst7_start))
        else:
            # OpenMM-format positions (Quantity in nm) → Angstroms → RST7
            pos_ang = self._openmm_positions_to_angstrom(positions)
            box_ang = self._openmm_box_to_angstrom(box_vectors)
            self._write_rst7(str(rst7_start), pos_ang, box_ang)

        suffix = traj_out.suffix.lower()
        is_nc_out = suffix in (".nc", ".ncdf")
        is_ascii_out = suffix in (".mdcrd", ".crd", ".trj")
        trajectory_format = self._trajectory_format_for_suffix(suffix)

        # ── Write Amber input file ────────────────────────────────────────
        in_file = workdir / f"_md_{run_index}.in"
        self._write_input(
            str(in_file),
            steps,
            stride,
            trajectory_format=trajectory_format,
        )

        # ── Output files (Amber native) ───────────────────────────────────
        mdout_file = workdir / f"_md_{run_index}.mdout"
        rst7_end = workdir / f"_end_{run_index}.rst7"

        if is_nc_out or is_ascii_out:
            native_traj = traj_out
        else:
            native_traj = workdir / f"_md_{run_index}.nc"

        # ── Set CUDA device and library path ──────────────────────────────
        env = os.environ.copy()
        uses_cuda = self._uses_cuda()
        if uses_cuda:
            # device_index >= 0 is a local-backend GPU slot; a negative sentinel
            # means the scheduler already bound CUDA_VISIBLE_DEVICES for this
            # array task — inherit it rather than pinning every task to device 0.
            if device_index >= 0:
                env["CUDA_VISIBLE_DEVICES"] = str(device_index)
            libnvjitlink_dir = _find_libnvjitlink_dir()
            if libnvjitlink_dir:
                existing_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{libnvjitlink_dir}:{existing_path}".rstrip(":")
                    if existing_path
                    else libnvjitlink_dir
                )
        cmd = [
            self.amber_executable,
            "-O",
            "-i",
            str(in_file),
            "-o",
            str(mdout_file),
            "-p",
            str(self.topology_file),
            "-c",
            str(rst7_start),
            "-r",
            str(rst7_end),
            "-x",
            str(native_traj),
            "-inf",
            "/dev/null",
        ]
        if uses_cuda:
            cmd.append("-AllowSmallBox")

        cmd.extend(self.amber_extra_args)

        try:
            subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=md_subprocess_timeout(),
            )
        except subprocess.CalledProcessError as exc:
            logging.error(
                "AmberEngine run %d failed (exit %d):\n%s",
                run_index,
                exc.returncode,
                exc.stderr,
            )
            return False
        except subprocess.TimeoutExpired:
            logging.error(
                "AmberEngine run %d timed out after %ss.",
                run_index,
                md_subprocess_timeout(),
            )
            return False
        except FileNotFoundError:
            logging.error(
                "AmberEngine: executable %r not found on PATH.", self.amber_executable
            )
            return False

        # ── Convert .nc → .xtc if needed ─────────────────────────────────
        if not (is_nc_out or is_ascii_out):
            try:
                self._convert_nc_to_xtc(str(native_traj), str(traj_out))
            except Exception as exc:
                logging.error("AmberEngine: NC→XTC conversion failed: %s", exc)
                return False

        # ── Clean up scratch files ────────────────────────────────────────
        cleanup_paths = [rst7_start, in_file, rst7_end, mdout_file]
        if not (is_nc_out or is_ascii_out):
            cleanup_paths.append(native_traj)

        for path in cleanup_paths:
            if path.exists():
                path.unlink()

        return True

    # ------------------------------------------------------------------
    # Input / output helpers
    # ------------------------------------------------------------------

    def _write_input(
        self,
        filepath: str,
        steps: int,
        stride: int,
        trajectory_format: str | None = None,
    ) -> None:
        """Write an Amber MD input (``.in``) file.

        If ``amber_input_file`` was set, the file is read as a format-string
        template with the following named replacements:

        ``{steps}``, ``{dt}``, ``{temp}``, ``{stride}``, ``{ntp}``, ``{ntb}``
        """
        ntp = 1 if self.npt else 0
        ntb = 2 if self.npt else 1
        trajectory_format = trajectory_format or self._resolved_trajectory_format()
        ioutfm = 0 if trajectory_format == "ascii" else 1
        pres_line = (
            f"  pres0={self.pressure:.4f}, barostat=2, taup=2.0," if self.npt else ""
        )

        if self.amber_input_file is not None:
            template_path = Path(self.amber_input_file)
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Amber input template not found: {self.amber_input_file}"
                )
            template = template_path.read_text()
            if "ioutfm" in template:
                template = re.sub(r"ioutfm\s*=\s*[01]", f"ioutfm={ioutfm}", template)
            elif "ioutfm" not in template:
                if "/" in template:
                    parts = template.split("/")
                    parts[-2] = parts[-2] + f"  ioutfm={ioutfm},\n"
                    template = "/".join(parts)
            if "ntxo" not in template:
                if "/" in template:
                    parts = template.split("/")
                    parts[-2] = parts[-2] + "  ntxo=2,\n"
                    template = "/".join(parts)
            content = template.format(
                steps=steps,
                dt=self.dt,
                temp=self.temperature,
                stride=stride,
                ntp=ntp,
                ntb=ntb,
            )
        else:
            content = (
                "Trails-MD MD run\n"
                " &cntrl\n"
                f"  imin=0, irest=0, ntx=1,\n"
                f"  nstlim={steps}, dt={self.dt:.6f},\n"
                f"  ntc=2, ntf=2,\n"
                # tempi must equal temp0: a spawned frame carries no velocities
                # (ntx=1, irest=0), so without tempi Amber generates velocities
                # at the default tempi=0 K and every walker records a cold-start
                # heat-up transient that pollutes the CV/MSM data. OpenMM and
                # GROMACS both start at the target temperature.
                f"  tempi={self.temperature:.2f}, temp0={self.temperature:.2f},\n"
                f"  ntt=3, gamma_ln=1.0,\n"
                f"  ig=-1,\n"
                f"  ntb={ntb}, ntp={ntp},{pres_line}\n"
                f"  cut=9.0,\n"
                f"  ntpr={steps}, ntwx={stride}, iwrap=1,\n"
                f"  ioutfm={ioutfm}, ntxo=2,\n"
                " /\n"
            )

        Path(filepath).write_text(content)

    def _resolved_trajectory_format(self) -> str:
        return resolve_amber_trajectory_format(
            self.amber_trajectory_format,
            self.amber_executable,
        )

    def _uses_cuda(self) -> bool:
        return "cuda" in Path(self.amber_executable).name.lower()

    def _trajectory_format_for_suffix(self, suffix: str) -> str:
        if suffix in (".mdcrd", ".crd", ".trj"):
            return "ascii"
        if suffix in (".nc", ".ncdf"):
            return "netcdf"
        return self._resolved_trajectory_format()

    def _convert_nc_to_xtc(self, nc_file: str, xtc_file: str) -> None:
        """Convert an Amber NetCDF trajectory to XTC using MDAnalysis."""
        import MDAnalysis as mda  # type: ignore

        u = mda.Universe(str(self.topology_file), nc_file)
        try:
            with mda.Writer(xtc_file, n_atoms=u.atoms.n_atoms) as writer:
                for _ts in u.trajectory:
                    writer.write(u)
        finally:
            u.trajectory.close()

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_rst7(
        filepath: str,
        positions_ang: np.ndarray,
        box_ang: np.ndarray | None = None,
    ) -> None:
        """Write an Amber RST7 restart file (coordinates only, no velocities).

        Parameters
        ----------
        filepath:
            Destination file path.
        positions_ang:
            Array of shape ``(natoms, 3)`` in Angstroms.
        box_ang:
            Array ``[a, b, c, alpha, beta, gamma]`` in Angstroms / degrees, or
            just ``[a, b, c]`` (angles default to 90°).  Pass ``None`` for
            vacuum (non-periodic) systems.
        """
        natoms = len(positions_ang)
        with open(filepath, "w") as fh:
            fh.write("Trails-MD restart\n")
            fh.write(f"{natoms:5d}  0.0000000e+00\n")
            flat = positions_ang.flatten()
            for i in range(0, len(flat), 6):
                chunk = flat[i : i + 6]
                fh.write("".join(f"{v:12.7f}" for v in chunk) + "\n")
            if box_ang is not None:
                a, b, c = float(box_ang[0]), float(box_ang[1]), float(box_ang[2])
                alpha = float(box_ang[3]) if len(box_ang) > 3 else 90.0
                beta = float(box_ang[4]) if len(box_ang) > 4 else 90.0
                gamma = float(box_ang[5]) if len(box_ang) > 5 else 90.0
                fh.write(
                    f"{a:12.7f}{b:12.7f}{c:12.7f}"
                    f"{alpha:12.7f}{beta:12.7f}{gamma:12.7f}\n"
                )

    @staticmethod
    def _split_start_state(start_coords):
        """Split *start_coords* into (positions, box_vectors).

        Mirrors the helper in ``OpenMMEngine`` so both engines share the same
        walker-dict interface.
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
        # Fallback: assume values are already in nm
        return (
            np.array([[v[0], v[1], v[2]] for v in positions], dtype=np.float64) * 10.0
        )

    @staticmethod
    def _openmm_box_to_angstrom(box_vectors) -> np.ndarray | None:
        """Convert an OpenMM box-vectors tuple to ``[a, b, c, α, β, γ]`` in
        Angstroms/degrees, handling triclinic cells (see
        :func:`trails_md.engines.base.box_vectors_to_abc_angles`)."""
        try:
            return box_vectors_to_abc_angles(box_vectors)
        except (TypeError, IndexError) as exc:
            logging.warning("AmberEngine: could not parse box vectors (%s).", exc)
            return None
