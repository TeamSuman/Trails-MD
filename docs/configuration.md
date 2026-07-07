# Configuration reference

A run is described by a single YAML file (paths are resolved relative to it).
All sections are validated by Pydantic; unknown or invalid values are rejected
at startup. Below, only non-obvious defaults are noted — see
`trails_md/config.py` for the authoritative schema.

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
    Set the `TRAILS_MD_TIMEOUT` environment variable (seconds) to guard
    against hung GROMACS/Amber subprocesses.

## `spawning`

| Key | Default | Description |
| --- | --- | --- |
| `spawn_scheme` | `density` | `density` \| `voronoi` \| `lof` \| `fps`. |
| `spawn_type` | `hard` | `hard` or `probabilistic`. |
| `search_mode` | `explore` | `explore` or `target` (toward `target`). |
| `walker` / `step` / `stride` | `10` / `10000` / `100` | Walkers per iteration / MD steps / save interval. |
| `max_workers` | `4` | Concurrent walkers (local backend). |
| `voronoi_clusters` | `150` | Voronoi cell count (`spawn_scheme: voronoi`). |
| `target` | — | CV-space target `[x, y, …]` when `search_mode: target`. |
| `recent_density_window` | `5` | Bins sampled in the last N iterations are down-weighted (`density`). |
| `lof_neighbors` | `20` | Neighbours for the LOF spawner. |
| `voronoi_periodic` | `false` | Wrap Voronoi cells periodically. |
| `voronoi_grid_size` | `250` | Grid resolution for Voronoi cell-area estimation. |
| `voronoi_max_clusters` | `5000` | Upper bound on auto-grown Voronoi cells. |
| `resolution_check_patience` | `5` | Iterations of bin-occupancy stall before refining the grid. |
| `resolution_max_bins` | `150` | Upper bound on per-axis bins when auto-refining. |
| `convergence_patience` | `0` | Bin-occupancy stall patience (legacy convergence). |

!!! note "Walker timeout (local backend)"
    `execution.walker_timeout` (seconds) kills a walker that runs longer than the
    limit and marks the batch failed — a guard against a hung in-process OpenMM
    walker. Off by default.

## `space_mode` and adaptive model

`space_mode`: `fixed` \| `pca` \| `tica` \| `tvae` \| `deep-tica`. For learned
modes:

| Key | Default | Description |
| --- | --- | --- |
| `adaptive_feature_type` | `distances` | `distances` \| `fitted_coords` \| `phi_psi`. **Note:** `phi_psi` is currently specific to the AIB9 peptide (it expects 9 `resname AIB` residues) and will raise on other systems — use `distances` or `fitted_coords` for general systems. |
| `retrain_freq` | `1` | Retrain the CV every N iterations. |
| `aggregate_memory` | `true` | Pool historical frames when retraining. |
| `max_adaptive_memory_frames` | `50000` | Cap on pooled frames. |
| `adaptive_model.lagtime` | `5` | Lag time for time-lagged CVs. |
| `adaptive_model.latent_dim` | `2` | CV dimensionality. |
| `adaptive_model.epochs` / `learning_rate` | `50` / `5e-4` | Training. |
| `adaptive_model.encoder_hidden_dims` | `[256,128]` | Network width. |

## `execution`

Where walkers run (workstation vs HPC). See [Execution](execution.md).

| Key | Default | Description |
| --- | --- | --- |
| `backend` | `local` | `local` \| `slurm` \| `pbs`. |
| `partition` / `account` / `walltime` | `None` / `None` / `01:00:00` | Scheduler resources. |
| `cpus_per_task` / `gpus_per_task` / `memory` | `1` / `0` / `None` | Per-walker resources. |
| `max_retries` | `1` | Resubmit failed walkers up to N times. |
| `poll_interval` / `submit_timeout` | `30` / `60` | Polling / command timeouts (s). |
| `max_in_flight` | `None` | Cap concurrent array elements (SLURM `%N`); set for large batches. |
| `wait_timeout` | `None` | Ceiling (s) on waiting for one array job before cancel; `None` derives from `walltime`. |
| `marker_grace` | `30` | Seconds to keep re-checking result markers after the job leaves the queue (shared-FS lag). |
| `module_loads` / `extra_directives` | `[]` / `[]` | `module load` lines / raw scheduler directives. |

See [Execution](execution.md) and [HPC scaling](hpc_scaling.md) for scaling and
fault-tolerance guidance.

## Top-level

| Key | Default | Description |
| --- | --- | --- |
| `outdir` | `runs/sampler_output` | Output directory. |
| `random_seed` | `42` | Global seed. |
| `checkpoint_freq` | `1` | Checkpoint every N iterations. `0` disables checkpointing (logged loudly at startup). |
| `min_success_fraction` | `1.0` | Fraction of walkers that must succeed to proceed; `< 1.0` tolerates transient failures (see [Execution](execution.md)). |
| `adaptive_angle_encoding` | `raw` | `raw` or `sincos`. Use `sincos` for periodicity-safe dihedral (`phi_psi`) features — strongly recommended whenever an angle can cross ±π. |
| `n_bins` / `min_values` / `max_values` | `[30,30]` / — / — | Binning for coverage / fixed-space bounds. |

!!! tip "GROMACS grompp strictness"
    `engine.gromacs_grompp_maxwarn` (default `0`, strict) controls how many
    grompp warnings are tolerated. grompp warnings often flag real problems (net
    charge, atom/molecule-name or coordinate/topology mismatches); raise it only
    after vetting the warnings.
