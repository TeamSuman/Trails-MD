# Changelog

All notable changes to Trails-MD are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

This cycle turns Trails-MD from a coverage-driven adaptive sampler into an
**MSM-convergence-driven** framework, hardens the engineering foundation, and
adds first-class **HPC scalability** and **VAMP-2 feature optimisation**. It also
adds a flux-weighted **transition-matrix convergence** gate with
**uncertainty-guided spawning**, and opt-in **landscape-adaptive binning**.

### Correctness review & HPC feature-test expansion (this pass)

Fixes from a fresh scientific/HPC review, plus a broadened HPC validation suite.

- **MSM segmentation (critical, silent):** `_collect_msm_trajectories` sliced the
  cumulative projection by a constant `step//stride`, which crosses walker
  boundaries for GROMACS (which writes the `t=0` frame, `step//stride + 1` frames)
  and any variable-length walker — injecting spurious inter-walker transitions
  into the MSM count matrix. It now segments by the stored per-walker frame
  records (exact counts) and ignores lower-dimensional projections.
- **Spawner ↔ frame-record misalignment (critical, silent):** the spawner pooled
  historical frames filtered by projection dimension while the core built the
  trajectory/frame-record lists unfiltered, so a spawn index could resolve to the
  wrong conformation when history mixed dimensionalities (e.g. a 2-D
  initial-trajectory injection alongside an n-D adaptive space). A single shared
  `pooled_history_iterations` helper now drives both, keeping them index-synced;
  occupancy tracking gained the same dimension guard.
- **Reproducibility:** core sampling draws (initial-walker replication,
  feature-memory pruning) moved off the global `random` module to an
  instance-bound generator on `SeedManager`, checkpointed for deterministic resume.
- **HPC oversubscription:** GROMACS `mdrun -ntomp` and the OpenMM CPU platform's
  thread count are derived from the walker's CPU allocation
  (`OMP_NUM_THREADS`/`SLURM_CPUS_PER_TASK`/`OPENMM_CPU_THREADS`) when not set
  explicitly; OpenMM OpenCL/HIP now get per-device isolation (not just CUDA).
- **GROMACS scratch hygiene:** `grompp`/`mdrun` no longer leave `mdout.mdp` /
  `state.cpt` / `state_prev.cpt` in the process working directory (the submit dir
  on HPC); they are redirected into the per-walker workdir and cleaned up.
- **HPC test suite:** expanded from OpenMM+fixed+density to a broad feature matrix
  (engines, spawners, learned CVs, MSM, feature selection, adaptive binning,
  resume, path reconstruction) with a **local-backend mirror** runner
  (`hpc_tests/run_local_matrix.sh`) so features can be validated off-cluster,
  feature-aware result validators, and a `RUNBOOK.md`.
- **Docs:** corrected `docs/cli.md` (`--log-level` default `WARNING`, `--config`
  default, `--ignore-missing-history`, `trails-md-path` batch flags) and stale
  "no chunking / space-unsafe manifest" claims in `docs/hpc_scaling.md` and
  `hpc_tests/DEBUGGING.md` (array chunking and the TAB-delimited manifest are
  implemented). Removed a hardcoded developer `gmx` path (now `TRAILS_MD_GMX`).

### HPC-scale review & hardening (this pass)

Fixes and features from a code-review pass focused on large-scale atomistic MD on
CPU/GPU HPC clusters (SLURM/PBS). Full findings: `docs/code_review.md`.

- **SLURM poller fix (critical):** `squeue --array` output (`<jobid>_<taskid>`)
  was never matched by the "job active?" check, so the driver gave up before
  walkers finished and marked them all failed. Now matched line-anchored.
- **Scheduler robustness:** poll-command timeouts are caught (no longer crash the
  campaign); an overall `wait_timeout` cancels a held/unschedulable job instead
  of hanging forever; a non-empty job id is required after submit; `max_in_flight`
  throttles concurrent array elements (SLURM `%N`); `marker_grace` absorbs
  shared-FS metadata lag; `scancel`/`qdel` cancellation hooks added.
- **Scheduler scaling:** `max_array_size` splits a batch larger than the
  scheduler's array-size cap (SLURM `MaxArraySize`, PBS `max_array_size`) into
  sequential sub-arrays; the array manifest is tab-delimited so task/result paths
  containing spaces survive the field split.
- **Deterministic resume:** the Python/NumPy/torch RNG state is checkpointed and
  restored (torch state stored as NumPy so the checkpoint unpickles without
  torch), so a resumed run reproduces an uninterrupted run's spawn/training draws.
- **Delta-checkpoint integrity:** each checkpoint records its delta chain
  (`history_chain.json`); `reconstruct_history` loudly reports a broken chain (a
  delta pruned/lost after the fact) instead of silently returning partial history.
- **GPU binding on schedulers:** array tasks inherit the scheduler's
  `CUDA_VISIBLE_DEVICES` (via a `WalkerTask.device_index = -1` sentinel) instead
  of every task pinning to device 0; OpenMM degrades to CPU on any device-load
  error instead of crashing the iteration.
- **Partial-failure tolerance:** `min_success_fraction` (default `1.0`, unchanged
  behaviour) lets long campaigns continue with the surviving walkers.
- **Checkpoint durability & resume correctness:** the `format_version` completion
  marker is now consulted everywhere, so a torn (crash-mid-save) checkpoint is
  never chosen for resume; `_atomic_pickle` `fsync`s before rename; `torch.load`
  uses `map_location="cpu"`; the checkpoint is written **last** in an iteration so
  a resumed `iter_N` reflects the completed iteration's convergence / resolution /
  MSM state; `torch` is now imported lazily.
- **Engine correctness:** Amber walkers set `tempi=temp0` (no more 0 K cold-start
  transient); triclinic boxes are converted correctly (no orthorhombic collapse)
  for Amber/GROMACS re-seeding; GROMACS `grompp -maxwarn` is configurable and
  strict (`0`) by default; deep-TICA projects in `eval()` mode.
- **Science:** opt-in `adaptive_angle_encoding: sincos` for periodicity-safe
  dihedral (`phi_psi`) CV features.
- **Config validation:** `checkpoint_freq` (0 disables, logged loudly),
  `gpus_per_task`, `max_in_flight`, `min_success_fraction`, `adaptive_angle_encoding`.
- **HPC test suite:** `hpc_tests/` — SLURM + PBS × CPU + GPU end-to-end validation
  with preflight, result validator, and a debugging playbook (`DEBUGGING.md`).
- **Docs:** `docs/hpc_scaling.md` (execution model, limits, and a WESTPA
  comparison/roadmap); `docs/code_review.md` (full findings & SOTA roadmap).

### Added

#### MSM convergence engine (Phase 1)
- New `trails_md/msm/` subsystem built on `deeptime`:
  - `MSMEstimator` — clustering (k-means / regular-space) on the CV/latent
    space → transition counts → MLE **or** Bayesian MSM → implied timescales,
    VAMP-2 score, PCCA+ metastable states, stationary distribution.
  - `diagnostics.py` — serialisable implied-timescale / Chapman-Kolmogorov /
    VAMP results.
  - `ConvergenceMonitor` with composable, pluggable criteria: implied-timescale
    stability, VAMP-2 plateau, stationary-distribution drift, Bayesian
    statistical-error thresholds, and a **flux-weighted transition-matrix**
    criterion (analytic Dirichlet `T_ij` uncertainty) — combined with
    `all` / `any` + patience.
- `MSMSpawner` (`spawn_scheme: msm`) — **uncertainty × leverage × flux**
  microstate seeding (`π_i · |ψ_i| · σ_out,i + α/√c_i`) on the estimator's shared
  clustering, throwing runs at the transitions whose in/out rates are uncertain
  and important; least-counts fallback before the first MSM. Knobs
  `msm.spawn_alpha` / `spawn_leverage` / `spawn_uncertainty`; `msm.stable_clustering`
  keeps microstate IDs comparable across iterations.
- **Weighted-ensemble resampling** — a real `WeightedEnsemble` split/merge core
  (Huber & Kim, weight-conserving) and `WESpawner` (`spawn_scheme: we`,
  `we_target_per_bin`), replacing the former placeholder.
- `MSMConfig` — all MSM behaviour is **opt-in** (`msm.enabled: false` by
  default), so existing configs are unaffected.
- Per-iteration MSM results are written to the run directory and checkpointed
  for resume.

#### MSM analysis & plotting
- `trails_md/analysis/` — matplotlib-free data utilities (`load_msm_series`,
  `load_latest_msm`, `load_cv_points`, free energies, free-energy surface) plus
  `plots` (implied timescales, VAMP-2 / timescale convergence, free-energy
  surface, metastable free energies, MSM network) and a one-command
  `trails-md-analyze` CLI producing a multi-panel convergence report.
- `msm.npz` now also stores the implied-timescale sweep and metastable
  populations for plotting.

#### Extensible ML collective variables (Phase 1)
- `trails_md/spaces/registry.py` — single source of truth for CV methods,
  their backends, and availability. Adds **VAMPNet** and **SPIB** (State
  Predictive Information Bottleneck) alongside TICA, TVAE, PCA, deep-TICA, and
  deep-LDA, with actionable errors when an optional backend is missing.

#### HPC execution backends (Phase 3)
- `trails_md/execution/` — pluggable `ExecutionBackend` (factory pattern):
  - `local` — multiprocessing across CPU/GPU slots on one node (multi-GPU
    workstation); preserves the original GPU-slot scheduling.
  - `slurm` / `pbs` — one **array job per iteration**, with completion driven by
    filesystem result markers and automatic **resubmission** of failed walkers
    (`execution.max_retries`).
- `ExecutionConfig` — backend selection plus scheduler resources (partition /
  queue, account, walltime, cpus/gpus per task, memory, module loads, extra
  directives). Defaults to `local`.

#### VAMP-2 input-feature selection
- `trails_md/spaces/feature_selection.py` — `vamp2_score`, `rank_candidates`,
  `greedy_vamp_selection`, and `FeatureSelector`: choose and **adaptively
  update** the input features that best resolve the slow dynamics.
- `FeatureSelectionConfig` (`feature_selection.enabled`, opt-in) — re-selects
  feature columns every `cadence` iterations; selection persisted for resume.

#### Landscape-adaptive binning
- `trails_md/binning/adaptive.py` — `AdaptiveBinner` + `BinnerFactory` with
  `gradient` (equi-resistance: fine bins where the density is low / barriers),
  `mab` (Minimal-Adaptive-Binning style front footholds), and `eigenvector` (bin
  along the leading slow CV coordinate) schemes alongside `uniform`. Selected via
  `binning.scheme`; wired into the density and weighted-ensemble spawners; opt-in,
  default `uniform` reproduces the constant-width grid exactly.

#### Adaptive CV quality & reproducibility (Phase 4)
- **VAMP-2-driven adaptive retraining** (`retrain_policy: vamp_adaptive`):
  `RetrainController` retrains the CV only when its VAMP-2 score on fresh data
  drops by more than `vamp_retrain_tol` below its reference (with
  `retrain_min_interval` / `retrain_max_interval` bounds). Reference score is
  checkpointed. The default `fixed` policy preserves the legacy schedule.
- **Reproducibility:** `SeedManager` now also seeds PyTorch Lightning
  (`seed_everything`, used by deep-TICA/LDA) and documents determinism limits.
- **Feature-type selection:** `feature_selection.candidate_feature_types` ranks
  whole feature types (`distances` / `fitted_coords` / `phi_psi`) by VAMP-2 and
  switches the loop to the best one (re-running column selection on a change).

#### End-user input file & tutorial
- **`trails-md-init`** writes a fully-annotated starter input file
  (`trails_md/templates.py`, mirrored to `examples/template.yaml`) covering
  every section, method choice, and hyperparameter. Documented in
  `docs/input_file.md`.
- **Jupyter notebook tutorial** with rendered plots
  (`examples/notebooks/adaptive_msm_tutorial.ipynb`): input file, VAMP-2 feature
  selection, MSM estimation, convergence, weighted ensemble, and the analysis
  report — all on fast synthetic data.

#### Tooling & docs
- Test suite (`pytest`) covering MSM, CV methods, execution backends, feature
  selection, config, spawners, and hardening (see the "Suite 95 → 118" note
  below for the current count).
- GitHub Actions CI (Python 3.10 / 3.11) running ruff + pytest.
- `ruff` / `black` / `isort` config, `.pre-commit-config.yaml`, `CONTRIBUTING.md`.
- `docs/` site (MkDocs) and tutorials; `CHANGELOG.md`; example run scripts for
  local, SLURM, and PBS.

### Changed (Phase 2 — engineering foundation)
- Refactored the per-iteration UI out of `core.py` into
  `trails_md/reporting.py` (`IterationReporter`); de-duplicated project-file
  CV loading.
- Migrated configuration to **Pydantic v2** (`field_validator` /
  `model_validator`, `model_dump()`); pinned `pydantic>=2.0`.
- Added **checkpoint format versioning** and generalised torch-encoder
  snapshots to tvae / vampnet / spib.
- Made `trails_md.spaces` import lazily so lightweight modules (e.g. the CV
  registry) import without MDAnalysis / torch.

### Production-readiness hardening (pre-release pass)
- **Reproducibility:** threaded the configured seed into all learned-CV training
  (SPIB no longer hardcodes `seed=42`; the torch RNG is reseeded before every
  `fit`), and `SeedManager` now requests deterministic torch algorithms.
- **Robustness:** the local backend tolerates a single walker's failure (and adds
  an opt-in `execution.walker_timeout` hang guard) instead of aborting the batch;
  delta-checkpoint resume reconstructs the full history (fixing a truncated
  `trails-md-path`), writes atomically, and tolerates a corrupt delta; fixed a
  target-mode spawn crash and a deep-TICA device mismatch.
- **Packaging:** OpenMM is now an optional, lazily-imported backend so the base
  `pip install` resolves; full PyPI metadata + single-sourced version;
  `CITATION.cff`; tag-driven PyPI release workflow.
- **CI/quality:** lint the whole tree (was a hand-picked subset), test on Python
  3.10–3.12, build the docs in CI.
- **Docs/examples/tests:** a self-contained CPU-only alanine-dipeptide
  hello-world; example configs for SPIB / deep-TICA / WE / target / PBS; an
  examples index; an API reference, references/citations page, and full CLI
  reference; +23 tests (delta checkpoint, reproducibility, timeout, spawners,
  paths). Suite 95 → 118, then +14 HPC-review regression tests
  (`tests/test_hpc_review_fixes.py`).

### Fixed (Phase 2)
- Renamed `AdaptiveSpaceModel.fited` → `fitted` (with a backwards-compatible
  loader for old checkpoints).
- Added MD subprocess timeouts (`TRAILS_MD_TIMEOUT`) for GROMACS / Amber.
- Validate trajectory files exist and are non-empty before CV extraction.
- Replaced hardcoded `/tmp` with `tempfile.gettempdir()`; narrowed an
  over-broad exception handler.
- Removed the dead `WEResampler` stub.

## [2.0.0] — baseline

Modular adaptive sampling framework: OpenMM / GROMACS / Amber engines, fixed or
learned (TICA / TVAE / PCA / deep-TICA) CV spaces, density / Voronoi / LOF / FPS
spawners, bin-occupancy convergence, checkpoint/restart, and lineage-aware path
reconstruction.
