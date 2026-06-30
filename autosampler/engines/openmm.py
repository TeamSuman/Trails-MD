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

        self.simulation = None
        self.positions = None

    @staticmethod
    def _available_platforms() -> list[str]:
        return [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]

    @classmethod
    def _get_platform(cls, platform_name: str):
        try:
            return Platform.getPlatformByName(platform_name)
        except Exception as exc:
            ", ".join(cls._available_platforms()) or "none"
            import logging
            logging.warning(f"OpenMM platform {platform_name} validation failed (likely because you are on a login node). Assuming compute nodes will have it. Error: {exc}")
            return None

    def prepare(
        self, conf: Path, top: Path, system_file: Path | None = None
    ) -> None:
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
        elif file_extension == ".crd":
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

        if self.npt:
            self.system.addForce(
                MonteCarloBarostat(
                    self.pressure, self.temperature, self.barostatInterval
                )
            )

        self.integrator.setConstraintTolerance(self.constraintTolerance)

    def _create_simulation(self, device_index: int):
        platform_props = {}
        if self.platform_name == "CUDA":
            platform_props = {
                "Precision": self.precision,
                "DeviceIndex": str(device_index),
            }
            try:
                self.simulation = Simulation(
                    self.topology,
                    self.system,
                    self.integrator,
                    self.platform,
                    platform_props,
                )
            except Exception as e:
                if "CUDA_ERROR_NO_DEVICE" not in str(e):
                    raise
                print("CUDA device unavailable; falling back to CPU platform.")
                self.platform = self._get_platform("CPU")
                self.platform_name = "CPU"
                self.simulation = Simulation(
                    self.topology, self.system, self.integrator, self.platform
                )
        else:
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, self.platform
            )
        self.simulation.context.setPositions(self.positions)

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
            self.simulation.context.setVelocitiesToTemperature(self.temperature)

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
            self.simulation.context.setVelocitiesToTemperature(self.temperature)

            self.simulation.minimizeEnergy()
            if self.should_equilibrate:
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
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
