import importlib.util
import inspect
import os
from pathlib import Path

from openmm import *
from openmm.app import *
from openmm.unit import *

from .base import MDEngine


class OpenMMEngine(MDEngine):
    """OpenMM specific implementation of MDEngine strategy."""

    def __init__(
        self,
        temperature: float = 300,
        pressure: float = 1.0,
        dt: float = 0.002,
        platform_name: str = "CUDA",
        precision: str = "mixed",
        npt: bool = False,
        equilibrate: bool = False,
        gromacs_include_dir: str | None = None,
        **kwargs,
    ):
        self.temperature_val = temperature
        self.pressure_val = pressure
        self.dt_val = dt
        self.platform_name = platform_name
        self.precision = precision
        self.npt = npt
        self.should_equilibrate = equilibrate
        self.gromacs_include_dir = gromacs_include_dir
        self.seed: int | None = kwargs.get("seed", None)

        self.simulation = None
        self.positions = None
        # When True, a persistent worker has cached this prepared engine and
        # re-arms it per walker instead of rebuilding the Context (see
        # ``_create_simulation`` and ``rearm_for_walker``). Off by default so the
        # normal one-shot path is completely unchanged.
        self._warm_reuse = False

    # Persistent-worker support: rebuilding the OpenMM Context (+ CUDA JIT) is the
    # dominant per-walker cost for short segments. This engine can instead keep a
    # warm Context alive across walkers and re-arm it, reproducing a fresh build
    # bit-for-bit (verified on the CPU platform). Subprocess engines (GROMACS,
    # Amber) gain nothing from this and leave the flag False.
    supports_warm_reuse = True

    def rearm_for_walker(self, seed: int | None) -> None:
        """Point an already-prepared, cached engine at the next walker.

        Only the per-walker state changes: the thermostat seed. Re-arming leaves
        the (JIT-warm) Context in place; ``_create_simulation`` then reseeds the
        integrator and reinitializes the Context so the run is identical to one
        started from a fresh build with this seed.
        """
        self.seed = seed

    @staticmethod
    def _available_platforms() -> list[str]:
        return [
            Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
        ]

    @classmethod
    def _get_platform(cls, platform_name: str):
        try:
            return Platform.getPlatformByName(platform_name)
        except Exception as exc:
            ", ".join(cls._available_platforms()) or "none"
            import logging

            logging.warning(
                f"OpenMM platform {platform_name} validation failed (likely because you are on a login node). Assuming compute nodes will have it. Error: {exc}"
            )
            return None

    def prepare(self, conf: Path, top: Path, system_file: Path | None = None) -> None:
        """Prepare the MD environment, e.g., setup system, topology, forces."""
        gro_file = str(conf)
        top_file = str(top)

        file_extension = os.path.splitext(gro_file)[1]
        top_extension = os.path.splitext(top_file)[1]

        # Parse inputs
        if file_extension == ".gro":
            self.gro = GromacsGroFile(gro_file)
            if top_extension in {".prmtop", ".parm7"}:
                self.top = AmberPrmtopFile(top_file)
            else:
                include_dir = self.gromacs_include_dir or os.path.dirname(top_file)
                self.top = GromacsTopFile(
                    top_file,
                    includeDir=include_dir,
                    periodicBoxVectors=self.gro.getPeriodicBoxVectors(),
                )
        elif file_extension == ".pdb":
            self.gro = PDBFile(gro_file)
            if top_extension in {".prmtop", ".parm7"}:
                self.top = AmberPrmtopFile(top_file)
            else:
                self.top = self.gro
        elif file_extension == ".xml":
            self.gro = XmlSerializer.deserialize(open(gro_file).read())
        elif file_extension in {".crd", ".rst7", ".ncrst", ".inpcrd"}:
            # tleap's `saveamberparm` writes .rst7 by default; accept the whole family.
            self.gro = AmberInpcrdFile(gro_file)
            self.top = AmberPrmtopFile(top_file, periodicBoxVectors=self.gro.boxVectors)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # System Configuration
        self.nonbondedMethod = PME
        self.nonbondedCutoff = 1.0 * nanometers
        self.ewaldErrorTolerance = 0.0005
        self.constraints = HBonds
        self.rigidWater = True
        self.constraintTolerance = 0.000001
        self.hydrogenMass = 1.5 * amu

        # Integration Options
        self.dt = self.dt_val * picoseconds
        self.temperature = self.temperature_val * kelvin
        self.friction = 1.0 / picosecond
        self.pressure = self.pressure_val * atmospheres
        self.barostatInterval = 25
        self.equilibrationSteps = 5000

        self.platform = self._get_platform(self.platform_name)

        self.topology = self.top.topology
        self.positions = self.gro.positions

        if system_file is not None:
            spec = importlib.util.spec_from_file_location(
                "openmm_system_module", str(system_file)
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load OpenMM system file: {system_file}")
            system_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(system_module)
            make_system_kwargs = {
                "temp": self.temperature.value_in_unit(kelvin),
                "dt": self.dt_val,
                "pressure": self.pressure_val,
            }
            signature = inspect.signature(system_module.make_system)
            accepted_kwargs = {
                key: value
                for key, value in make_system_kwargs.items()
                if key in signature.parameters
            }
            self.system, self.integrator = system_module.make_system(
                self.top,
                **accepted_kwargs,
            )
            # A user's make_system cannot know the per-walker seed, so thread it
            # here too. Without this the thermostat RNG of a custom-system_file run
            # is unseeded and walkers are NOT reproducible -- the deterministic-seed
            # guarantee silently held only for the built-in system path.
            if self.seed is not None and hasattr(self.integrator, "setRandomNumberSeed"):
                self.integrator.setRandomNumberSeed(self.seed)
        else:
            self.system = self.top.createSystem(
                nonbondedMethod=self.nonbondedMethod,
                nonbondedCutoff=self.nonbondedCutoff,
                constraints=self.constraints,
                rigidWater=self.rigidWater,
                ewaldErrorTolerance=self.ewaldErrorTolerance,
                hydrogenMass=self.hydrogenMass,
            )
            self.integrator = LangevinMiddleIntegrator(
                self.temperature, self.friction, self.dt
            )
            if self.seed is not None:
                self.integrator.setRandomNumberSeed(self.seed)

        if self.npt:
            barostat = MonteCarloBarostat(
                self.pressure, self.temperature, self.barostatInterval
            )
            if self.seed is not None:
                barostat.setRandomNumberSeed(self.seed)
            self.system.addForce(barostat)

        self.integrator.setConstraintTolerance(self.constraintTolerance)

    @staticmethod
    def _cpu_thread_count() -> int | None:
        """Thread cap for the OpenMM CPU platform, from the run environment.

        Without a cap the CPU platform uses every core on the node, so several
        CPU walkers sharing a node oversubscribe. Honour (in order)
        ``OPENMM_CPU_THREADS``, the scheduler's ``SLURM_CPUS_PER_TASK``, or
        ``OMP_NUM_THREADS``; return ``None`` (OpenMM decides) when none is set.
        """
        for var in ("OPENMM_CPU_THREADS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
            value = os.environ.get(var)
            if value and value.isdigit() and int(value) > 0:
                return int(value)
        return None

    def _platform_properties(self, device_index: int) -> dict:
        """Platform-specific isolation properties.

        ``device_index >= 0`` is a local-backend GPU slot to pin to; a negative
        sentinel means "let the scheduler's ``*_VISIBLE_DEVICES`` decide"
        (SLURM/PBS array tasks) so tasks are not all pinned to physical device 0.
        """
        name = self.platform_name
        if name == "CUDA":
            props = {"Precision": self.precision}
            if device_index >= 0:
                props["DeviceIndex"] = str(device_index)
            return props
        if name == "OpenCL":
            props = {"OpenCLPrecision": self.precision}
            if device_index >= 0:
                props["OpenCLDeviceIndex"] = str(device_index)
            return props
        if name == "HIP":
            return {"HipDeviceIndex": str(device_index)} if device_index >= 0 else {}
        if name == "CPU":
            threads = self._cpu_thread_count()
            return {"Threads": str(threads)} if threads else {}
        return {}

    def _new_simulation(self, platform_props: dict):
        if self.platform is None:  # login-node validation returned no platform
            return Simulation(self.topology, self.system, self.integrator)
        return Simulation(
            self.topology,
            self.system,
            self.integrator,
            self.platform,
            platform_props or {},
        )

    def _create_simulation(self, device_index: int):
        import logging

        # Warm path: a persistent worker has already built this Context on this
        # device. Re-arm it instead of rebuilding. Reseeding the integrator and
        # then reinitializing the Context reproduces a fresh build bit-for-bit,
        # because the thermostat RNG is re-derived from the new seed on
        # reinitialize (a naive reseed without reinitialize does NOT — verified).
        if self._warm_reuse and self.simulation is not None:
            if self.seed is not None and getattr(self, "integrator", None) is not None:
                self.integrator.setRandomNumberSeed(self.seed)
            self.simulation.context.reinitialize(preserveState=False)
            self.simulation.context.setPositions(self.positions)
            return

        platform_props = self._platform_properties(device_index)
        try:
            self.simulation = self._new_simulation(platform_props)
        except Exception as e:
            message = str(e)
            cuda_device_error = self.platform_name == "CUDA" and any(
                token in message
                for token in (
                    "CUDA_ERROR_NO_DEVICE",
                    "could not be loaded",
                    "no CUDA-capable device",
                    "invalid device",
                )
            )
            if cuda_device_error:
                # Any device-load failure (no device, bad index, driver mismatch)
                # should degrade to CPU rather than kill the walker/iteration.
                logging.warning(
                    "OpenMM CUDA device unavailable (%s); falling back to CPU.",
                    message,
                )
                self.platform = self._get_platform("CPU")
                self.platform_name = "CPU"
                self.simulation = self._new_simulation(
                    self._platform_properties(device_index)
                )
            elif platform_props:
                # OpenCL/HIP/CPU: a rejected isolation property must never be
                # worse than before — retry without the extra properties.
                logging.warning(
                    "OpenMM platform properties %s rejected (%s); retrying "
                    "without them.",
                    platform_props,
                    message,
                )
                self.simulation = self._new_simulation({})
            else:
                raise
        self.simulation.context.setPositions(self.positions)

    def _write_gpu_binding_marker(
        self, traj_out: Path, requested_device_index: int, run_index: int
    ) -> None:
        """Record which platform/device this walker actually ran on.

        Written as ``<traj_out>.gpu.json`` next to the trajectory so GPU device
        isolation can be verified after the fact, independent of the run's log
        level and identically for the local and scheduler backends. Capturing the
        *resolved* platform also exposes a silent CUDA→CPU fallback (a bad device
        pin) that would otherwise pass unnoticed. See
        ``hpc_tests/checks/validate_results.py`` (``GPU_BINDING``).
        """
        import json
        import logging

        name = self.platform_name
        device_index = ""
        device_name = ""
        if name in ("CUDA", "OpenCL", "HIP") and self.platform is not None:
            idx_prop = (
                "OpenCLDeviceIndex"
                if name == "OpenCL"
                else ("HipDeviceIndex" if name == "HIP" else "DeviceIndex")
            )
            name_prop = "OpenCLDeviceName" if name == "OpenCL" else "DeviceName"
            for prop, dest in ((idx_prop, "index"), (name_prop, "name")):
                try:
                    value = self.platform.getPropertyValue(
                        self.simulation.context, prop
                    )
                except Exception:  # noqa: BLE001 - property may be unsupported
                    value = ""
                if dest == "index":
                    device_index = value
                else:
                    device_name = value
        visible = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("GPU_DEVICE_ORDINAL")
            or ""
        )
        marker = {
            "run_index": int(run_index),
            "platform": name,
            "requested_device_index": int(requested_device_index),
            "device_index": device_index,
            "device_name": device_name,
            "visible_devices": visible,
        }
        try:
            Path(f"{traj_out}.gpu.json").write_text(json.dumps(marker))
        except OSError as exc:
            logging.warning("Could not write GPU binding marker: %s", exc)
        logging.info(
            "Trails-MD walker GPU binding: run=%s platform=%s device_index=%s "
            "visible_devices=%s device=%s",
            run_index,
            name,
            device_index or requested_device_index,
            visible,
            device_name,
        )

    def run_production(
        self,
        run_index: int,
        start_coords: Path,
        steps: int,
        traj_out: Path,
        stride: int,
        device_index: int,
    ) -> bool:
        """Execute a production run."""
        self._create_simulation(device_index)
        self._write_gpu_binding_marker(traj_out, device_index, run_index)

        traj_out_str = str(traj_out)
        if os.path.exists(traj_out_str):
            os.remove(traj_out_str)

        start_positions, start_box_vectors = self._split_start_state(start_coords)
        if start_positions is not None:
            # We assume start_coords is a file containing positions or a state that OpenMM can load
            # This is a simplification; in reality, we need to extract coords from start_coords.
            # Assuming start_coords is passed as an object containing positions if it's not a Path
            # Actually, `start_coords` should be positions directly, or we parse it.
            # For backward compatibility, let's allow it to be the positions directly.
            if start_box_vectors is not None:
                self.simulation.context.setPeriodicBoxVectors(*start_box_vectors)
            self.simulation.context.setPositions(start_positions)
            self.simulation.context.setVelocitiesToTemperature(
                self.temperature, self.seed if self.seed is not None else 0
            )

        self.simulation.reporters = [self._trajectory_reporter(traj_out_str, stride)]

        success = False
        try:
            self.simulation.currentStep = 0
            self.simulation.step(steps)
            success = True
        except Exception as e:
            print(f"Production run failed ({e}), attempting reinitialize+recovery...")
            self.simulation.reporters = []
            if os.path.exists(traj_out_str):
                os.remove(traj_out_str)
            self.simulation.context.reinitialize()
            recovery_positions = (
                start_positions if start_positions is not None else self.positions
            )
            if start_box_vectors is not None:
                self.simulation.context.setPeriodicBoxVectors(*start_box_vectors)
            self.simulation.context.setPositions(recovery_positions)
            self.simulation.context.setVelocitiesToTemperature(
                self.temperature, self.seed if self.seed is not None else 0
            )

            self.simulation.minimizeEnergy()
            if self.should_equilibrate:
                self.simulation.context.setVelocitiesToTemperature(
                    self.temperature, self.seed if self.seed is not None else 0
                )
                self.simulation.step(self.equilibrationSteps)

            self.simulation.reporters = [
                self._trajectory_reporter(traj_out_str, stride)
            ]
            self.simulation.currentStep = 0
            self.simulation.step(steps)
            success = True

        return success

    def _trajectory_reporter(self, traj_out: str, stride: int):
        return XTCReporter(traj_out, stride, enforcePeriodicBox=True)

    @staticmethod
    def _split_start_state(start_coords):
        if isinstance(start_coords, dict):
            return start_coords.get("positions"), start_coords.get("box_vectors")
        return start_coords, None
