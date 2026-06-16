from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("gpu_ids")
    @classmethod
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

    @field_validator("amber_trajectory_format")
    @classmethod
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

    @field_validator("lagtime", "latent_dim", "epochs")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("batch_size must be 'auto' or a positive integer")
            return value
        if value <= 0:
            raise ValueError("batch_size must be 'auto' or a positive integer")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be greater than 0")
        return value

    @field_validator("dropout_rate")
    @classmethod
    def validate_dropout_rate(cls, value: float) -> float:
        if value < 0 or value >= 1:
            raise ValueError("dropout_rate must be >= 0 and < 1")
        return value

    @field_validator("encoder_hidden_dims", "decoder_hidden_dims", "deep_tica_hidden_dims")
    @classmethod
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

    @field_validator("cadence", "lagtime", "n_microstates", "n_timescales")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    @field_validator("cluster_method")
    @classmethod
    def _cluster_method(cls, value: str) -> str:
        value = value.lower()
        if value not in {"kmeans", "regspace"}:
            raise ValueError("cluster_method must be 'kmeans' or 'regspace'")
        return value

    @field_validator("estimator")
    @classmethod
    def _estimator(cls, value: str) -> str:
        value = value.lower()
        if value not in {"mle", "bayesian"}:
            raise ValueError("estimator must be 'mle' or 'bayesian'")
        return value

    @field_validator("convergence_mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        value = value.lower()
        if value not in {"all", "any"}:
            raise ValueError("convergence_mode must be 'all' or 'any'")
        return value


class FeatureSelectionConfig(BaseModel):
    """VAMP-2 based selection/optimisation of the input features for the CV/MSM.

    Opt-in (``enabled=False`` by default). When enabled, the adaptive loop
    periodically scores feature columns by VAMP-2 and keeps the subset that best
    resolves the slow dynamics, adaptively updating it every ``cadence``
    iterations.
    """

    enabled: bool = False
    method: str = "greedy_vamp"  # "greedy_vamp" | "all"
    lagtime: int = 10
    cadence: int = 5  # re-select every N iterations (adaptive update)
    max_features: Optional[int] = None  # cap on selected columns/groups
    dim: Optional[int] = None  # singular values retained when scoring
    min_gain: float = 1e-4  # minimum VAMP-2 gain to add a feature group
    # Optional: rank these feature *types* by VAMP-2 and use the best one.
    # Empty -> always use the top-level `adaptive_feature_type`.
    candidate_feature_types: List[str] = []

    @field_validator("method")
    @classmethod
    def _method(cls, value: str) -> str:
        if value not in {"greedy_vamp", "all"}:
            raise ValueError("feature_selection.method must be 'greedy_vamp' or 'all'")
        return value

    @field_validator("candidate_feature_types")
    @classmethod
    def _candidate_types(cls, value: List[str]) -> List[str]:
        valid = {"distances", "fitted_coords", "phi_psi"}
        bad = [v for v in value if v not in valid]
        if bad:
            raise ValueError(
                "feature_selection.candidate_feature_types must be a subset of "
                f"{sorted(valid)}; got invalid {bad}"
            )
        return value

    @field_validator("lagtime", "cadence")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value


class ExecutionConfig(BaseModel):
    """Where and how walker MD jobs are dispatched.

    ``backend: local`` (default) runs walkers as local subprocesses across CPU
    or GPU slots (multi-GPU workstation). ``slurm`` / ``pbs`` submit each
    iteration's walkers as a scheduler array job for CPU-only or GPU HPC
    clusters. Scheduler fields are ignored by the local backend.
    """

    backend: str = "local"  # "local" | "slurm" | "pbs"
    # Scheduler resource requests (per array task = one walker).
    partition: Optional[str] = None  # SLURM partition / PBS queue
    account: Optional[str] = None
    walltime: str = "01:00:00"
    cpus_per_task: int = 1
    gpus_per_task: int = 0
    memory: Optional[str] = None  # e.g. "8G"
    # Robustness / polling.
    max_retries: int = 1  # resubmit failed walkers up to this many times
    poll_interval: float = 30.0  # seconds between scheduler polls
    submit_timeout: float = 60.0  # seconds for a submit/poll command
    module_loads: List[str] = []  # `module load ...` lines for job scripts
    extra_directives: List[str] = []  # raw #SBATCH / #PBS lines
    job_name: str = "autosampler"

    @field_validator("backend")
    @classmethod
    def _backend(cls, value: str) -> str:
        value = value.lower()
        if value not in {"local", "slurm", "pbs"}:
            raise ValueError("execution.backend must be 'local', 'slurm', or 'pbs'")
        return value

    @field_validator("cpus_per_task", "max_retries")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("poll_interval", "submit_timeout")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value


class AutoSamplerConfig(BaseModel):
    system: SystemConfig
    engine: EngineConfig
    spawning: SpawningConfig
    msm: MSMConfig = Field(default_factory=MSMConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    space_mode: str = "fixed"
    n_bins: List[int] = [30, 30]
    min_values: Optional[List[float]] = None
    max_values: Optional[List[float]] = None
    outdir: str = "runs/sampler_output"
    random_seed: int = 42
    checkpoint_freq: int = 1
    save_features: bool = True
    retrain_freq: int = 1
    # CV-retraining policy: "fixed" (every retrain_freq iters) or "vamp_adaptive"
    # (retrain when the CV's VAMP-2 score drops by vamp_retrain_tol).
    retrain_policy: str = "fixed"
    vamp_retrain_tol: float = 0.1
    retrain_min_interval: int = 1
    retrain_max_interval: Optional[int] = None
    aggregate_memory: bool = True
    max_adaptive_memory_frames: int = 50000
    adaptive_feature_type: str = "distances"
    adaptive_model: AdaptiveModelConfig = Field(default_factory=AdaptiveModelConfig)

    @field_validator("retrain_policy")
    @classmethod
    def _retrain_policy(cls, value: str) -> str:
        if value not in {"fixed", "vamp_adaptive"}:
            raise ValueError("retrain_policy must be 'fixed' or 'vamp_adaptive'")
        return value

    @field_validator("space_mode")
    @classmethod
    def validate_space_mode(cls, value: str) -> str:
        from autosampler.spaces.registry import FIXED_MODE, adaptive_modes

        valid = (FIXED_MODE,) + adaptive_modes()
        if value not in valid:
            raise ValueError(f"space_mode must be one of {valid}; got {value!r}")
        return value

    @model_validator(mode="before")
    @classmethod
    def promote_spawning_n_bins(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(values)
        if "n_bins" not in values:
            spawning = values.get("spawning")
            if isinstance(spawning, dict) and spawning.get("n_bins") is not None:
                values["n_bins"] = spawning["n_bins"]
        return values
