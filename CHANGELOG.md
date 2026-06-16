# Changelog

All notable changes to AutoSampler are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased] — `devel`

This cycle turns AutoSampler from a coverage-driven adaptive sampler into an
**MSM-convergence-driven** framework, hardens the engineering foundation, and
adds first-class **HPC scalability** and **VAMP-2 feature optimisation**.

### Added

#### MSM convergence engine (Phase 1)
- New `autosampler/msm/` subsystem built on `deeptime`:
  - `MSMEstimator` — clustering (k-means / regular-space) on the CV/latent
    space → transition counts → MLE **or** Bayesian MSM → implied timescales,
    VAMP-2 score, PCCA+ metastable states, stationary distribution.
  - `diagnostics.py` — serialisable implied-timescale / Chapman-Kolmogorov /
    VAMP results.
  - `ConvergenceMonitor` with composable, pluggable criteria: implied-timescale
    stability, VAMP-2 plateau, stationary-distribution drift, and Bayesian
    statistical-error thresholds (combined with `all` / `any` + patience).
- `MSMSpawner` (`spawn_scheme: msm`) — least-counts / MSM-uncertainty restart
  seeding that drives the MSM toward convergence on its slow processes.
- `MSMConfig` — all MSM behaviour is **opt-in** (`msm.enabled: false` by
  default), so existing configs are unaffected.
- Per-iteration MSM results are written to the run directory and checkpointed
  for resume.

#### Extensible ML collective variables (Phase 1)
- `autosampler/spaces/registry.py` — single source of truth for CV methods,
  their backends, and availability. Adds **VAMPNet** and **SPIB** (State
  Predictive Information Bottleneck) alongside TICA, TVAE, PCA, deep-TICA, and
  deep-LDA, with actionable errors when an optional backend is missing.

#### HPC execution backends (Phase 3)
- `autosampler/execution/` — pluggable `ExecutionBackend` (factory pattern):
  - `local` — multiprocessing across CPU/GPU slots on one node (multi-GPU
    workstation); preserves the original GPU-slot scheduling.
  - `slurm` / `pbs` — one **array job per iteration**, with completion driven by
    filesystem result markers and automatic **resubmission** of failed walkers
    (`execution.max_retries`).
- `ExecutionConfig` — backend selection plus scheduler resources (partition /
  queue, account, walltime, cpus/gpus per task, memory, module loads, extra
  directives). Defaults to `local`.

#### VAMP-2 input-feature selection
- `autosampler/spaces/feature_selection.py` — `vamp2_score`, `rank_candidates`,
  `greedy_vamp_selection`, and `FeatureSelector`: choose and **adaptively
  update** the input features that best resolve the slow dynamics.
- `FeatureSelectionConfig` (`feature_selection.enabled`, opt-in) — re-selects
  feature columns every `cadence` iterations; selection persisted for resume.

#### Adaptive CV quality & reproducibility (Phase 4)
- **VAMP-2-driven adaptive retraining** (`retrain_policy: vamp_adaptive`):
  `RetrainController` retrains the CV only when its VAMP-2 score on fresh data
  drops by more than `vamp_retrain_tol` below its reference (with
  `retrain_min_interval` / `retrain_max_interval` bounds). Reference score is
  checkpointed. The default `fixed` policy preserves the legacy schedule.
- **Reproducibility:** `SeedManager` now also seeds PyTorch Lightning
  (`seed_everything`, used by deep-TICA/LDA) and documents determinism limits.

#### Tooling & docs
- Test suite (`pytest`) covering MSM, CV methods, execution backends, feature
  selection, config, spawners, and hardening — **49 tests**.
- GitHub Actions CI (Python 3.10 / 3.11) running ruff + pytest.
- `ruff` / `black` / `isort` config, `.pre-commit-config.yaml`, `CONTRIBUTING.md`.
- `docs/` site (MkDocs) and tutorials; `CHANGELOG.md`; example run scripts for
  local, SLURM, and PBS.

### Changed (Phase 2 — engineering foundation)
- Refactored the per-iteration UI out of `core.py` into
  `autosampler/reporting.py` (`IterationReporter`); de-duplicated project-file
  CV loading.
- Migrated configuration to **Pydantic v2** (`field_validator` /
  `model_validator`, `model_dump()`); pinned `pydantic>=2.0`.
- Added **checkpoint format versioning** and generalised torch-encoder
  snapshots to tvae / vampnet / spib.
- Made `autosampler.spaces` import lazily so lightweight modules (e.g. the CV
  registry) import without MDAnalysis / torch.

### Fixed (Phase 2)
- Renamed `AdaptiveSpaceModel.fited` → `fitted` (with a backwards-compatible
  loader for old checkpoints).
- Added MD subprocess timeouts (`AUTOSAMPLER_MD_TIMEOUT`) for GROMACS / Amber.
- Validate trajectory files exist and are non-empty before CV extraction.
- Replaced hardcoded `/tmp` with `tempfile.gettempdir()`; narrowed an
  over-broad exception handler.
- Removed the dead `WEResampler` stub.

## [2.0.0] — baseline

Modular adaptive sampling framework: OpenMM / GROMACS / Amber engines, fixed or
learned (TICA / TVAE / PCA / deep-TICA) CV spaces, density / Voronoi / LOF / FPS
spawners, bin-occupancy convergence, checkpoint/restart, and lineage-aware path
reconstruction.
