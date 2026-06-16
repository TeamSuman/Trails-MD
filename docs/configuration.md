# Configuration reference

A run is described by a single YAML file (paths are resolved relative to it).
All sections are validated by Pydantic; unknown or invalid values are rejected
at startup. Below, only non-obvious defaults are noted — see
`autosampler/config.py` for the authoritative schema.

## `system`

| Key | Default | Description |
| --- | --- | --- |
| `conf_file` | — | Coordinate file (`.gro`, `.pdb`, …). |
| `top_file` | — | Topology file. |
| `topology` | `amber` | `gromacs` \| `amber` \| `charmm`. |
| `system_file` | `None` | Optional Python module building a custom OpenMM `System`. |
| `project_file` | `None` | Python module with `extract_cvs(...)` for fixed CVs. |
| `trajectory_topology_file` | `None` | Topology used to read trajectories (defaults to `top_file`). |
| `feature_selection` | `protein and not (type H)` | MDAnalysis atom selection for features. |

## `engine`

| Key | Default | Description |
| --- | --- | --- |
| `md_engine` | `openmm` | `openmm` \| `gromacs` \| `amber`. |
| `platform_name` | `CUDA` | OpenMM platform (`CUDA`, `CPU`, `OpenCL`, `Reference`). |
| `precision` | `mixed` | OpenMM precision. |
| `temperature` / `pressure` / `dt` | `300` / `1.0` / `0.002` | Thermostat / barostat / timestep. |
| `npt` / `equilibrate` | `false` / `false` | Constant-pressure / pre-equilibration. |
| `gpu_ids` | `None` | Explicit GPU device ids for the local backend. |
| `gromacs_*` / `amber_*` | — | Engine-specific executables and `mdrun`/`pmemd` options. |

!!! tip "MD timeouts"
    Set the `AUTOSAMPLER_MD_TIMEOUT` environment variable (seconds) to guard
    against hung GROMACS/Amber subprocesses.

## `spawning`

| Key | Default | Description |
| --- | --- | --- |
| `spawn_scheme` | `density` | `density` \| `voronoi` \| `lof` \| `fps` \| `msm`. |
| `spawn_type` | `hard` | `hard` or `probabilistic`. |
| `search_mode` | `explore` | `explore` or `target` (toward `target`). |
| `walker` / `step` / `stride` | `10` / `10000` / `100` | Walkers per iteration / MD steps / save interval. |
| `max_workers` | `4` | Concurrent walkers (local backend). |
| `voronoi_clusters` | `150` | Cells / microstates (also used by the MSM spawner). |
| `convergence_patience` | `0` | Bin-occupancy stall patience (legacy convergence). |

## `space_mode` and adaptive model

`space_mode`: `fixed` \| `pca` \| `tica` \| `tvae` \| `vampnet` \| `spib` \|
`deep-tica` \| `deep-lda`. For learned modes:

| Key | Default | Description |
| --- | --- | --- |
| `adaptive_feature_type` | `distances` | `distances` \| `fitted_coords` \| `phi_psi`. |
| `retrain_freq` | `1` | Retrain the CV every N iterations. |
| `aggregate_memory` | `true` | Pool historical frames when retraining. |
| `max_adaptive_memory_frames` | `50000` | Cap on pooled frames. |
| `adaptive_model.lagtime` | `5` | Lag time for time-lagged CVs. |
| `adaptive_model.latent_dim` | `2` | CV dimensionality. |
| `adaptive_model.epochs` / `learning_rate` | `50` / `5e-4` | Training. |
| `adaptive_model.encoder_hidden_dims` | `[256,128]` | Network width. |
| `adaptive_model.spib_n_states` / `spib_beta` | `10` / `1e-3` | SPIB knobs. |

## `feature_selection`

VAMP-2 input-feature selection (opt-in). See [Feature selection](feature_selection.md).

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Turn feature selection on. |
| `method` | `greedy_vamp` | `greedy_vamp` \| `all`. |
| `lagtime` | `10` | Lag time for VAMP-2 scoring. |
| `cadence` | `5` | Re-select every N iterations. |
| `max_features` | `None` | Cap on selected columns/groups. |
| `min_gain` | `1e-4` | Minimum VAMP-2 gain to keep adding features. |

## `msm`

Markov State Model estimation and convergence (opt-in). See [MSM](msm.md).

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Build an MSM each iteration and use MSM convergence. |
| `cadence` / `min_frames` | `1` / `1000` | MSM cadence / minimum frames before first MSM. |
| `lagtime` / `lagtimes` | `10` / `None` | MSM lag / implied-timescale sweep. |
| `n_microstates` / `cluster_method` | `100` / `kmeans` | Discretisation. |
| `estimator` | `mle` | `mle` or `bayesian` (error bars). |
| `n_timescales` / `n_metastable` | `3` / `None` | Slow processes / PCCA+ states. |
| `convergence_criteria` | ITS + VAMP-2 | List of `{name, params}`. |
| `convergence_mode` / `convergence_patience` | `all` / `2` | Combine criteria / patience. |

## `execution`

Where walkers run (workstation vs HPC). See [Execution](execution.md).

| Key | Default | Description |
| --- | --- | --- |
| `backend` | `local` | `local` \| `slurm` \| `pbs`. |
| `partition` / `account` / `walltime` | `None` / `None` / `01:00:00` | Scheduler resources. |
| `cpus_per_task` / `gpus_per_task` / `memory` | `1` / `0` / `None` | Per-walker resources. |
| `max_retries` | `1` | Resubmit failed walkers up to N times. |
| `poll_interval` / `submit_timeout` | `30` / `60` | Polling / command timeouts (s). |
| `module_loads` / `extra_directives` | `[]` / `[]` | `module load` lines / raw scheduler directives. |

## Top-level

| Key | Default | Description |
| --- | --- | --- |
| `outdir` | `runs/sampler_output` | Output directory. |
| `random_seed` | `42` | Global seed. |
| `checkpoint_freq` | `1` | Checkpoint every N iterations. |
| `n_bins` / `min_values` / `max_values` | `[30,30]` / — / — | Binning for coverage / fixed-space bounds. |
