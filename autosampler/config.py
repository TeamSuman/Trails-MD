from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class SystemConfig(BaseModel):
    conf_file: str
    top_file: str
    topology: str = "amber"
    system_file: Optional[str] = None
    project_file: Optional[str] = None
    trajectory_topology_file: Optional[str] = None
    feature_selection: str = "protein and not (type H)"


class EngineConfig(BaseModel):
    md_engine: str = "openmm"
    platform_name: str = "CUDA"
    precision: str = "mixed"
    npt: bool = False
    equilibrate: bool = False
    temperature: float = 300.0
    pressure: float = 1.0
    dt: float = 0.002
    gpu_ids: Optional[List[int]] = None
    gromacs_include_dir: Optional[str] = None
    gromacs_executable: str = "gmx"
    gromacs_mdrun_nb: Optional[str] = None
    gromacs_mdrun_pme: Optional[str] = None
    gromacs_mdrun_update: Optional[str] = None
    gromacs_mdrun_bonded: Optional[str] = None
    gromacs_mdrun_pin: Optional[str] = None
    gromacs_mdrun_ntmpi: int = 1
    gromacs_mdrun_ntomp: Optional[int] = None
    gromacs_mdrun_extra_args: List[str] = []
    amber_executable: str = "pmemd"
    amber_input_file: Optional[str] = None
    amber_extra_args: List[str] = []
    amber_trajectory_format: str = "auto"

    @validator("gpu_ids")
    def validate_gpu_ids(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        if not value:
            raise ValueError("gpu_ids must contain at least one device id")
        if any(device_id < 0 for device_id in value):
            raise ValueError("gpu_ids must be non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("gpu_ids must not contain duplicates")
        return value

    @validator("amber_trajectory_format")
    def validate_amber_trajectory_format(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", "netcdf", "ascii"}:
            raise ValueError(
                "amber_trajectory_format must be 'auto', 'netcdf', or 'ascii'"
            )
        return value


class SpawningConfig(BaseModel):
    spawn_scheme: str = "density"
    spawn_type: str = "hard"
    search_mode: str = "explore"
    n_bins: Optional[List[int]] = None
    walker: int = 10
    step: int = 10000
    stride: int = 100
    max_workers: int = 4
    target: Optional[List[float]] = None
    recent_density_window: int = 5
    voronoi_clusters: int = 150
    voronoi_periodic: bool = False
    voronoi_grid_size: int = 250
    lof_neighbors: int = 20
    resolution_check_patience: int = 5
    resolution_max_bins: int = 150
    voronoi_max_clusters: int = 5000
    convergence_patience: int = 0


class AdaptiveModelConfig(BaseModel):
    lagtime: int = 5
    latent_dim: int = 2
    epochs: int = 50
    batch_size: Union[int, str] = "auto"
    learning_rate: float = 0.0005
    encoder_hidden_dims: List[int] = [256, 128]
    decoder_hidden_dims: List[int] = [128, 256]
    dropout_rate: float = 0.1
    deep_tica_hidden_dims: List[int] = [256, 128]
    # SPIB (State Predictive Information Bottleneck) hyperparameters.
    spib_n_states: int = 10
    spib_beta: float = 1e-3

    @validator("lagtime", "latent_dim", "epochs")
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @validator("batch_size")
    def validate_batch_size(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("batch_size must be 'auto' or a positive integer")
            return value
        if value <= 0:
            raise ValueError("batch_size must be 'auto' or a positive integer")
        return value

    @validator("learning_rate")
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be greater than 0")
        return value

    @validator("dropout_rate")
    def validate_dropout_rate(cls, value: float) -> float:
        if value < 0 or value >= 1:
            raise ValueError("dropout_rate must be >= 0 and < 1")
        return value

    @validator("encoder_hidden_dims", "decoder_hidden_dims", "deep_tica_hidden_dims")
    def validate_hidden_dims(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("hidden dimension lists must not be empty")
        if any(dim <= 0 for dim in value):
            raise ValueError("hidden dimensions must be greater than 0")
        return value


class MSMConfig(BaseModel):
    """Configuration for Markov State Model estimation and MSM-based convergence.

    All MSM behaviour is opt-in: with ``enabled=False`` (the default) the
    adaptive loop keeps its legacy bin-occupancy convergence and no MSM is built,
    so existing configs and examples are unaffected.
    """

    enabled: bool = False
    # How often (in iterations) to (re)estimate the MSM. 1 = every iteration.
    cadence: int = 1
    # Minimum cumulative frames before the first MSM is attempted.
    min_frames: int = 1000
    lagtime: int = 10
    # Optional lag-time ladder for an implied-timescale sweep (diagnostics).
    lagtimes: Optional[List[int]] = None
    n_microstates: int = 100
    cluster_method: str = "kmeans"  # "kmeans" | "regspace"
    estimator: str = "mle"  # "mle" | "bayesian"
    n_bayesian_samples: int = 50
    n_timescales: int = 3
    n_metastable: Optional[int] = None
    # Convergence: list of {name, params} criteria combined with all/any.
    convergence_criteria: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"name": "implied_timescales", "params": {"tol": 0.1, "n_timescales": 2}},
            {"name": "vamp2", "params": {"tol": 0.05}},
        ]
    )
    convergence_mode: str = "all"  # "all" | "any"
    convergence_patience: int = 2

    @validator("cadence", "lagtime", "n_microstates", "n_timescales")
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @validator("cluster_method")
    def _cluster_method(cls, value: str) -> str:
        value = value.lower()
        if value not in {"kmeans", "regspace"}:
            raise ValueError("cluster_method must be 'kmeans' or 'regspace'")
        return value

    @validator("estimator")
    def _estimator(cls, value: str) -> str:
        value = value.lower()
        if value not in {"mle", "bayesian"}:
            raise ValueError("estimator must be 'mle' or 'bayesian'")
        return value

    @validator("convergence_mode")
    def _mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {"all", "any"}:
            raise ValueError("convergence_mode must be 'all' or 'any'")
        return value


class AutoSamplerConfig(BaseModel):
    system: SystemConfig
    engine: EngineConfig
    spawning: SpawningConfig
    msm: MSMConfig = Field(default_factory=MSMConfig)
    space_mode: str = "fixed"
    n_bins: List[int] = [30, 30]
    min_values: Optional[List[float]] = None
    max_values: Optional[List[float]] = None
    outdir: str = "runs/sampler_output"
    random_seed: int = 42
    checkpoint_freq: int = 1
    save_features: bool = True
    retrain_freq: int = 1
    aggregate_memory: bool = True
    max_adaptive_memory_frames: int = 50000
    adaptive_feature_type: str = "distances"
    adaptive_model: AdaptiveModelConfig = Field(default_factory=AdaptiveModelConfig)

    @validator("space_mode")
    def validate_space_mode(cls, value: str) -> str:
        from autosampler.spaces.registry import FIXED_MODE, adaptive_modes

        valid = (FIXED_MODE,) + adaptive_modes()
        if value not in valid:
            raise ValueError(f"space_mode must be one of {valid}; got {value!r}")
        return value

    @root_validator(pre=True)
    def promote_spawning_n_bins(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(values)
        if "n_bins" not in values:
            spawning = values.get("spawning")
            if isinstance(spawning, dict) and spawning.get("n_bins") is not None:
                values["n_bins"] = spawning["n_bins"]
        return values
