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
| `topology` | `amber` | Reserved metadata field; the OpenMM engine infers the input format from file extensions, so this is currently a no-op. |
| `system_file` | `None` | Optional Python module building a custom OpenMM `System`. |
| `project_file` | `None` | Python module with `extract_cvs(...)` for fixed CVs. |
| `trajectory_topology_file` | `None` | Topology used to read trajectories (defaults to `top_file`). |
| `feature_selection` | `protein and not (type H)` | MDAnalysis atom selection for features. |

## `engine`

| Key | Default | Description |
| --- | --- | --- |
| `md_engine` | `openmm` | `openmm` \| `gromacs` \| `amber`. |
| `platform_name` | `CUDA` | OpenMM platform (`CUDA`, `OpenCL`, `HIP`, `CPU`; `Reference` = unaccelerated fallback). |
| `precision` | `mixed` | OpenMM precision. |
| `temperature` / `pressure` / `dt` | `300` / `1.0` / `0.002` | Thermostat / barostat / timestep. |
| `npt` / `equilibrate` | `false` / `false` | Constant-pressure / pre-equilibration. |
| `gpu_ids` | `None` | Explicit GPU device ids for the local backend. |
| `seed` | `None` | Deterministic engine RNG seed (integrator/thermostat/barostat). `None` derives it from the run's `random_seed`. |
| `gromacs_executable` / `gromacs_include_dir` | `gmx` / `None` | GROMACS binary and force-field include dir. **`gromacs_include_dir` is also required when OpenMM reads a GROMACS `.top`** that `#include`s a force field (e.g. `amber99sb.ff/...`). |
| `gromacs_mdrun_nb/pme/update/bonded/pin` | `None` | `mdrun` offload placement (`cpu`/`gpu`/`auto`). |
| `gromacs_mdrun_ntmpi` / `gromacs_mdrun_ntomp` | `1` / `None` | Thread-MPI ranks / OpenMP threads. |
| `gromacs_mdrun_extra_args` / `gromacs_grompp_maxwarn` | `[]` / `0` | Extra `mdrun` flags / grompp warning tolerance. |
| `amber_executable` / `amber_input_file` | `pmemd` / `None` | Amber binary and custom mdin. |
| `amber_trajectory_format` | `auto` | `auto` \| `netcdf` \| `ascii`. |
| `amber_extra_args` | `[]` | Extra `pmemd` flags. |

!!! tip "MD timeouts"
    Set the `TRAILS_MD_TIMEOUT` environment variable (seconds) to guard
    against hung GROMACS/Amber subprocesses.

## `spawning`

| Key | Default | Description |
| --- | --- | --- |
| `spawn_scheme` | `density` | `density` \| `voronoi` \| `lof` \| `fps` \| `we` (weighted ensemble; the [kinetics-mode](modes.md) spawner) \| `msm` (MSM-guided; needs `msm.enabled`). |
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
| `we_target_per_bin` | `4` | Walkers kept per occupied bin by the `we` spawner. |
| `inherit_velocities` | `false` | **Kinetics mode.** Continue the parent's velocities instead of resampling. Requires `spawn_scheme: we` **and** `md_engine: openmm`. |
| `recycle_target` | `None` | **Kinetics mode.** Source→sink recycling box, one `[lo, hi]` per CV dimension; recycled weight per τ gives `MFPT = 1/flux` (Hill relation). Requires `spawn_scheme: we`; must be bounded in every CV dimension. See [Exploration vs. kinetics](modes.md). |
| `recycle_basis_index` | `0` | Frame index recycled walkers restart from (the source). |

!!! note "Walker timeout (local backend)"
    `execution.walker_timeout` (seconds) kills a walker that runs longer than the
    limit and marks the batch failed — a guard against a hung in-process OpenMM
    walker. Off by default.

## `space_mode` and adaptive model

`space_mode`: `fixed` \| `pca` \| `tica` \| `tvae` \| `deep-tica` (+ experimental
`vampnet` \| `spib` \| `deep-lda`; see [Collective variables](cv_methods.md)). For
learned modes:

| Key | Default | Description |
| --- | --- | --- |
| `adaptive_feature_type` | `distances` | `distances` \| `fitted_coords` \| `phi_psi`. **Note:** `phi_psi` is currently specific to the AIB9 peptide (it expects 9 `resname AIB` residues) and will raise on other systems — use `distances` or `fitted_coords` for general systems. |
| `retrain_freq` | `1` | Retrain the CV every N iterations (used when `retrain_policy: fixed`). |
| `retrain_policy` | `fixed` | `fixed` (retrain every `retrain_freq`) or `vamp_adaptive` (retrain when the VAMP-2 score drops by more than `vamp_retrain_tol`). |
| `vamp_retrain_tol` | `0.1` | Relative VAMP-2 drop that triggers a retrain (`vamp_adaptive`). |
| `retrain_min_interval` / `retrain_max_interval` | `1` / `None` | Floor/ceiling on iterations between adaptive retrains. |
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
| `persistent_workers` | `false` | **Local backend:** keep worker processes and warm OpenMM `Context`s alive across iterations instead of rebuilding each walker. Large speed-up for short segments (OpenMM only). |
| `partition` / `account` / `walltime` | `None` / `None` / `01:00:00` | Scheduler resources. |
| `cpus_per_task` / `gpus_per_task` / `memory` | `1` / `0` / `None` | Per-walker resources. |
| `gres` | `None` | SLURM `--gres` per array element (e.g. `gpu:1`). Use with `gpus_per_task: 0` on sites whose GPU partition requires `--gres`/`--gpus` and rejects `--gpus-per-task`. |
| `max_retries` | `1` | Resubmit failed walkers up to N times. |
| `submit_retry_limit` / `submit_retry_interval` | `20` / `15` | Retries + backoff (s) for *transient* submit rejections (per-user QOS/association submit-job caps, scheduler rate limits); permanent errors still fail fast. |
| `poll_interval` / `submit_timeout` | `30` / `60` | Polling / command timeouts (s). |
| `max_in_flight` | `None` | Cap concurrent array elements (SLURM `%N`); set for large batches. |
| `max_array_size` | `None` | Split batches larger than this into sub-arrays (beat SLURM `MaxArraySize` / PBS `max_array_size`). |
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
| `save_features` | `true` | Write the extracted input features (`features.npz`) per iteration. |
| `system.initial_trajectory` | `None` | Seed the sampling history from an existing trajectory instead of the single input structure. |
| `adaptive_angle_encoding` | `raw` | `raw` or `sincos`. Use `sincos` for periodicity-safe dihedral (`phi_psi`) features — strongly recommended whenever an angle can cross ±π. |
| `n_bins` / `min_values` / `max_values` | `[30,30]` / — / — | Binning for coverage / fixed-space bounds. |

!!! tip "GROMACS grompp strictness"
    `engine.gromacs_grompp_maxwarn` (default `0`, strict) controls how many
    grompp warnings are tolerated. grompp warnings often flag real problems (net
    charge, atom/molecule-name or coordinate/topology mismatches); raise it only
    after vetting the warnings.
