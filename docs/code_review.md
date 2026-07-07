# Trails-MD code review — HPC-scale correctness, robustness & SOTA roadmap

Scope: a code-review pass over Trails-MD in the context of running **scalable,
large-scale atomistic MD on HPC clusters** (CPU and GPU nodes, SLURM and PBS),
plus a scientific-correctness review of the CV / spawning / MSM subsystems, and a
roadmap toward state-of-the-art competitiveness.

Legend: **[FIXED]** = addressed in this pass (see `CHANGELOG.md`); **[OPEN]** =
documented here with a recommended fix but not yet implemented (larger change or
needs cluster/GPU validation).

---

## Executive summary

The framework is well-structured (clean strategy/factory abstractions for
engines, spawners, CV methods, execution backends; opt-in MSM/feature-selection;
a real test suite). The most serious issues were concentrated in three areas:

1. **The SLURM backend did not actually work on a real cluster** — the "is the
   array job still running?" poll never matched `squeue --array` output, so every
   iteration was declared finished before any walker completed. **[FIXED]**
2. **Silent scientific-correctness bugs** — dihedral CVs fed as raw radians
   (periodicity torn at ±π), Amber walkers starting at 0 K, triclinic boxes
   collapsed to orthorhombic, GROMACS emitting an extra t=0 frame.
3. **Resume/robustness gaps for long campaigns** — torn checkpoints treated as
   complete, a single walker failure aborting a multi-day run, checkpoint written
   mid-iteration, RNG state not persisted.

This pass fixed the highest-impact, verifiable items and added an HPC test suite
(`hpc_tests/`) plus regression tests (`tests/test_hpc_review_fixes.py`, +14).
Remaining **[OPEN]** items are catalogued below with concrete fixes.

---

## 1. HPC execution & scheduling (`trails_md/execution/`)

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 1.1 | Critical | `SlurmBackend._job_active` matched `\b<jobid>\b`, which never matches `squeue --array` output `12345_0` (`_` is a word char) → poller thinks a running array is gone → all walkers marked failed. `slurm.py`. | **[FIXED]** line-anchored `^\s*<jobid>(?:_\|\b)` match + regression test. |
| 1.2 | High | No overall wait deadline: a held/unschedulable array (`H`/`Q`) hangs the campaign forever. `scheduler.py:_wait_for_completion`. | **[FIXED]** `wait_timeout` (derived from `walltime` if unset) cancels via `scancel`/`qdel`. |
| 1.3 | High | Poll/submit `subprocess.TimeoutExpired` was uncaught → a slow `squeue` at scale crashes the run. `scheduler.py`. | **[FIXED]** poll timeouts caught + retried; submit timeout wrapped. |
| 1.4 | High | Scheduler array tasks left `device_index=0`, and engines forced `CUDA_VISIBLE_DEVICES=0` / `DeviceIndex=0`, clobbering the scheduler's GPU allocation → all co-located walkers pile on GPU 0. `execution/base.py`, engines. | **[FIXED]** `device_index=-1` sentinel = "inherit scheduler binding"; engines only pin when `>= 0`. |
| 1.5 | High | No array chunking: >`MaxArraySize` (SLURM 1001) / PBS `max_array_size` walkers/iteration is rejected outright. | **[OPEN]** chunk `pending` into ≤ limit sub-arrays; expose `max_array_size`. Mitigated by `max_in_flight`. |
| 1.6 | Med | No concurrency cap → a huge array floods the scheduler and triggers a stat-storm each poll. | **[FIXED]** `max_in_flight` → `--array=0-N%M`. |
| 1.7 | Med | 2 s grace after job exit is too short for NFS/Lustre marker lag → successful walkers misreported failed. | **[FIXED]** configurable `marker_grace` (default 30 s) with re-checks. |
| 1.8 | Med | PBS backend is OpenPBS/PBS-Pro-only (`-J`, `PBS_ARRAY_INDEX`) but was documented as Torque-compatible. | **[FIXED docs]** clarified; Torque (`-t`/`PBS_ARRAYID`) variant is **[OPEN]**. |
| 1.9 | Med | PBS does not export the submit env by default and no `#PBS -V` is emitted → bare `pmemd`/`gmx` not found. | **[FIXED docs]** call out `module_loads` / `#PBS -V`; auto-`-V` is **[OPEN]**. |
| 1.10 | Med | Manifest split on a single space breaks on `outdir` paths containing spaces. `scheduler.py:_render_script`. | **[OPEN]** use NUL/tab delimiter; documented in `hpc_tests/DEBUGGING.md`. |
| 1.11 | Low | Empty/garbage job id after submit went undetected. | **[FIXED]** submit now requires a parseable job id. |
| 1.12 | Low | Local backend indexes results by `task.index` assuming contiguous 0..n-1. `local.py`. | **[OPEN]** map by position/dict (latent; safe today). |

Architecture note: for very large, sustained fan-out the array-per-iteration
model is fundamentally overhead-bound and ceiling-bound; see
[HPC scaling & WESTPA comparison](hpc_scaling.md) for the persistent-worker-pool
roadmap.

---

## 2. MD engines (`trails_md/engines/`)

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 2.1 | High (silent physics) | Amber default input used `irest=0, ntx=1` with no `tempi` → velocities generated at **0 K**; every walker records a heat-up transient (OpenMM/GROMACS start at target T). `amber.py`. | **[FIXED]** `tempi=temp0` + regression test. |
| 2.2 | High (silent physics) | Triclinic boxes (truncated octahedron / rhombic dodecahedron — the norm) collapsed to orthorhombic on re-seeding → wrong volume/density/PBC. `amber.py`/`gromacs.py`. | **[FIXED]** shared `box_vectors_to_abc_angles` (correct α,β,γ) + round-trip test. |
| 2.3 | High (silent physics) | GROMACS writes a t=0 frame → `step//stride + 1` frames/walker vs `step//stride` for OpenMM/Amber, desyncing the fixed-frame MSM/CV segmentation. `gromacs.py`. | **[OPEN]** drop the t=0 frame or segment by actual per-trajectory counts (see §4.x). |
| 2.4 | High (HPC crash) | OpenMM `DeviceIndex` used absolute GPU ids but OpenMM numbers relative to `CUDA_VISIBLE_DEVICES`; the fallback only caught `CUDA_ERROR_NO_DEVICE`, so a bad index re-raised and killed the iteration. `openmm.py`. | **[FIXED]** broadened fallback to CPU; scheduler binding via 1.4; local restricted-visibility documented. |
| 2.5 | Med-High | `-AllowSmallBox` passed unconditionally on `pmemd.cuda` disables the too-small-box safety abort → silent min-image errors. `amber.py`. | **[OPEN]** gate behind an opt-in flag. |
| 2.6 | Med | `random_seed` never propagated to integrators/thermostats/barostats/velocity generation (OpenMM `setRandomNumberSeed`/`setVelocitiesToTemperature(seed)`, GROMACS `gen_seed`, Amber `ig`). | **[OPEN]** thread the seed (+per-walker offset) for reproducible runs. |
| 2.7 | Med | OpenMM device isolation only for CUDA; OpenCL/HIP and CPU walkers not isolated → oversubscription. `openmm.py`. | **[OPEN]** set `OpenCLDeviceIndex`/`HipDeviceIndex`/CPU `Threads`. |
| 2.8 | Med | GROMACS `grompp -maxwarn 5` silently masked real setup errors (net charge, name/count mismatch). | **[FIXED]** configurable `gromacs_grompp_maxwarn`, default `0`; grompp stderr surfaced. |
| 2.9 | Med | Amber template `ioutfm`/`ntxo` injected via naive `split("/")` → breaks on `&end`/multi-namelist templates. `amber.py`. | **[OPEN]** parse the `&cntrl` block or require `{}` placeholders. |
| 2.10 | Med | GROMACS mdrun: no per-walker `-pinoffset`, no default `-ntomp` cap → concurrent local walkers oversubscribe cores. | **[OPEN]** derive `-ntomp` from cores/concurrency, distinct `-pinoffset` per slot. |
| 2.11 | Med | OpenMM default path hardcodes HMR (1.5 amu) + PME + HBonds; HMR silently alters kinetics/timescales, and PME is wrong for implicit-solvent/vacuum (only the custom `system_file` escapes). | **[OPEN]** make HMR/constraints/nonbonded configurable; auto-select non-periodic method. |
| 2.12 | Low | Amber cutoff 9 Å vs OpenMM/GROMACS 10 Å; hand-written RST7 field widths overflow >100k atoms; `XTCReporter(enforcePeriodicBox=True)` on non-periodic systems. | **[OPEN]** unify cutoff; write RST7 via ParmEd/MDAnalysis; gate `enforcePeriodicBox` on PBC. |

---

## 3. Checkpointing, resume & the core loop (`checkpoints/`, `core.py`, `config.py`)

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 3.1 | Critical | The `format_version` completion marker was written "to signal completeness" but **no reader consulted it** → a torn (crash-mid-save) `iter_N` was chosen for resume, then `load` read torn/absent files. `checkpoints/manager.py`. | **[FIXED]** all scanners + `load` now gate on the marker; regression test. |
| 3.2 | Critical | Delta checkpointing makes each `history.pkl` non-self-contained; deleting/losing one dir silently drops that slice of history forever (breaks lineage/MSM). `manager.py`. | **[OPEN]** keep history self-contained (delta only large `features`), or write a verified delta-chain manifest. |
| 3.3 | High | A single failed walker aborted the entire multi-day campaign (`core.py` + `cli.py`), defeating the execution layer's per-walker fault tolerance. | **[FIXED]** `min_success_fraction` (default 1.0 = legacy) drops failed walkers and continues. |
| 3.4 | High | Resume is non-deterministic: numpy/torch/Python RNG state is never checkpointed, so a resumed run diverges from an uninterrupted one despite the "exact deterministic restart" claim. | **[OPEN]** persist/restore RNG state (or derive a per-iteration seed = `hash(base, iter)`). |
| 3.5 | High | Checkpoint was written **mid-iteration** (before occupancy/resolution/convergence/MSM updates), so resume silently reverted an adaptive `n_bins` bump or a `converged` flag. `core.py`. | **[FIXED]** save moved to the end of the iteration (post-processing). |
| 3.6 | High | Unbounded RAM/disk growth: `feature_memory` appended every iteration and never pruned; `history` keeps every iteration's `features`/`frames`/`projection`; `features` stored twice (npz + history). | **[OPEN]** bound `feature_memory` to the fit window; store `features` only in `features.npz` and reference it. |
| 3.7 | Med | `_atomic_pickle` did not `fsync` before rename → node power-loss/FS failover can still leave a torn file. `manager.py`. | **[FIXED]** `flush()` + `os.fsync()` before `os.replace`. |
| 3.8 | Med | `checkpoint_freq=0` silently disabled all checkpointing (a walltime kill loses everything). | **[FIXED]** allowed but logged loudly at startup; negatives rejected. |
| 3.9 | Med | `torch.load` without `map_location` fails restoring a GPU checkpoint on a CPU node. `manager.py`. | **[FIXED]** `map_location="cpu"`; `torch` import made lazy. |
| 3.10 | Med | Pickle-based checkpoints are library-version fragile and unsafe; no migration logic or recorded lib versions. | **[OPEN]** record lib versions + warn; prefer explicit serialization for reconstructable pieces. |
| 3.11 | Med | Spawner cumulative pool filters history by projection-dim while core's frame-record pool does not → index offset → spawning from the **wrong** frame (silent) when a history entry's dim differs (e.g. `initial_trajectory` physical CVs vs latent). `density.py` vs `core.py`. | **[OPEN]** build both pools from one shared, identically-filtered function (or return `(iter,walker,frame)` keys). |
| 3.12 | Low | Config gaps: `walltime`/`memory` free strings validated only at submit; `convergence_criteria` unschematized; leftover debug `print`. | **[PARTIAL]** debug print removed; added validators; format validation is **[OPEN]**. |

---

## 4. Scientific / algorithmic (`spaces/`, `spawners/`, `msm/`, `binning/`)

| # | Sev | Finding | Status |
| --- | --- | --- | --- |
| 4.1 | Critical (silent science) | Dihedral angles fed to PCA/TICA/VAE/k-means as **raw radians** — no sin/cos. θ=+179° and −179° become maximally distant, tearing basins that cross ±π (the flagship AIB9 φ/ψ workflow). `spaces/features.py`, `model.py`, `scalers.py`. | **[FIXED, opt-in]** `adaptive_angle_encoding: sincos` + `encode_angles_sincos`; **strongly recommend making it default** after validation. |
| 4.2 | High | Fixed-space grid binning / FPS / LOF / density use plain Euclidean distance on periodic CVs (only `VoronoiBinner` has a periodic option) → artefactual seams at ±π. `binning/spatial.py`, `spawners/fps.py`,`lof.py`. | **[OPEN]** add minimum-image distance + wrapped bins, gated on a per-CV periodic flag. |
| 4.3 | Med-High | "SPIB" omits the defining self-consistent state-refinement loop; labels are a one-shot k-means → it is a time-lagged VAE predicting fixed bins, not SPIB. `spaces/spib.py`. | **[OPEN]** implement the label-update loop, or rename the method. |
| 4.4 | Med | Deep-TICA projected in **train mode** (no `.eval()`) → dropout/BatchNorm make projections batch-dependent. `spaces/model.py`. | **[FIXED]** `.eval()` before projecting (matches tvae/vampnet/spib). |
| 4.5 | Med | `batch_size="auto"` = whole dataset → full-batch training (only `epochs` updates); TVAE loader `shuffle=False`. | **[OPEN]** default a real minibatch (256–1024), shuffle the TVAE loader. |
| 4.6 | Med | Learned-CV **sign/degeneracy not canonicalized** across retrains; mostly masked by per-retrain reprojection, but a fixed latent `target` points at the opposite region after a sign flip. `model.py`. | **[OPEN]** pin sign (largest-loading positive) or align to previous eigenvectors; warn on fixed latent targets. |
| 4.7 | Med | Cross-iteration MSM convergence compares across **different clusterings** when `stable_clustering=False` (default) → conflates "converged" with "clustering jitter". `msm/convergence.py`. | **[OPEN]** default `stable_clustering=True` for cross-iteration criteria, or share one clustering. |
| 4.8 | Med | MSM estimated with **unweighted** counts on non-equilibrium adaptive/WE data; WE weights are discarded, TICA unreweighted → biased π. `msm/estimator.py`, `spawners/we.py`. | **[OPEN]** feed WE/importance weights to a weighted estimator (or TRAM); document π validity. |
| 4.9 | Low-Med | Global-`np.random` spawners (density/voronoi/lof/fps) vs per-call `default_rng` (msm/we); resume restores neither → non-reproducible resumed spawns (see 3.4). | **[OPEN]** give every spawner an explicit seeded `Generator`, checkpoint its state. |
| 4.10 | Low-Med | WE weight carry-over concentrates on survivors; new frames enter near 0 → the ensemble starves exactly the new regions adaptive sampling seeks. `spawners/we.py`. | **[OPEN]** seed new frames at per-bin/median weight; renormalize per occupied bin. |
| 4.11 | Low | MSM-guided Dirichlet uncertainty uses frame counts, not out-transition row counts (`count_matrix.sum(axis=1)`), mis-ranking "uncertain" transitions. `spawners/msm.py`. | **[OPEN]** use the connected count-matrix row sums (already on `MSMResult`). |
| 4.12 | Low | `extract_pairwise_distances` applies PBC min-image to *intramolecular* distances → wrapped (too-short) distances for extended conformers approaching L/2. `features.py`. | **[OPEN]** make the molecule whole, compute without `box`. |

---

## 5. SOTA roadmap — making Trails-MD globally competitive

Where the field is (adaptive sampling / MSM / ML-CV) and where Trails-MD should
go. Grouped by theme, roughly in priority order.

### 5.1 Scientific correctness first (prerequisite for competitiveness)
- Make **periodicity-safe CVs the default** (4.1/4.2): sin/cos for angular
  features, minimum-image distances and wrapped bins everywhere. No SOTA MSM
  result is trustworthy without this.
- **Reweighting for kinetics** (4.8): integrate TRAM / dTRAM / Koopman
  reweighting so MSMs built on adaptively/WE-biased data give unbiased
  equilibrium populations and rates — this is table stakes for publishable
  kinetics and where WESTPA+MSM/haMSM and pyEMMA/deeptime workflows already are.
- **CV canonicalization & uncertainty** (4.6): stable, sign-pinned CVs with
  bootstrap/Bayesian error bars on timescales (deeptime `BayesianMSM` is already
  wired — surface its errors in convergence and spawning).

### 5.2 Adaptive-sampling algorithms (the core value proposition)
- **Goal-oriented / rate-targeted adaptive sampling**: beyond coverage and
  least-counts, add mean-first-passage-time-reduction and transition-state
  (committor-gradient) seeding — the direction of REAP, AdaptiveBandit, and
  core-set/committor methods.
- **Reinforcement-learning / bandit spawners** (REAP, AdaptiveBandit): the bandit
  scaffolding (`msm.spawn_alpha`, leverage, uncertainty) is a good base; add a
  learned reward over CV importance.
- **Weighted ensemble done fully** (haMSM): couple the WE spawner to the MSM
  (history-augmented MSM / WE steady-state) for direct rate estimation — a natural
  fit given both subsystems already exist.
- **Committor / string / interface methods** as first-class spawners (FFS,
  milestoning, transition-path sampling) for mechanism, not just coverage.

### 5.3 ML collective variables (keep pace with the ML-CV frontier)
- Finish **SPIB** (4.3) and add modern CVs: **VAMPnets** (present — validate),
  **time-lagged / GraphVAMPnets**, **SchNet/e3nn-style equivariant featurizers**,
  and **transferable CVs**. Lean on `mlcolvar`/`deeptime` rather than
  reimplementing.
- **Auto-featurization** with periodic-aware, size-transferable descriptors
  (contacts, dihedral sin/cos, SOAP/ACSF) and the existing VAMP-2 feature
  selection promoted to a default-on, well-documented workflow.

### 5.4 HPC & performance (the stated goal)
- **Persistent worker-pool backend** (MPI/ZMQ) to remove per-iteration submit
  latency and array-size ceilings — the single biggest scaling win; see
  [HPC scaling](hpc_scaling.md).
- **Consolidated HDF5/Zarr storage** to end the thousands-of-small-files
  metadata-server pressure and make provenance a single artifact.
- **Async / overlapped iterations**: begin analysis/CV-training on completed
  walkers while stragglers finish, instead of a hard per-iteration barrier.
- **GPU-resident analysis** (featurization, TICA/clustering on GPU) for large
  systems; multi-GPU-per-walker for very large systems.
- **Checkpoint/resume hardening** (3.2/3.4/3.10): self-contained checkpoints, RNG
  state, version metadata — essential for month-long campaigns.

### 5.5 Usability, reproducibility & community
- **Reproducibility**: propagate the seed into MD engines (2.6) and persist RNG
  state (3.4); ship a `--dry-run`/cost estimate; record the full environment.
- **Provenance & standards**: export to community formats (MDTraj/MDAnalysis,
  H5MD; MSM objects to deeptime/PyEMMA) so results interoperate.
- **Restart-from-anything, monitoring**: a live dashboard (occupancy, implied
  timescales, VAMP-2, free-energy surface) and Prometheus-style metrics for long
  HPC runs.
- **Benchmarks & validation gallery**: alanine dipeptide, AIB9, a fast-folder
  (e.g. chignolin) with published reference kinetics, run in CI-lite, to prove
  correctness and competitiveness.
- **Packaging**: conda-forge recipe; pinned, tested environment per engine.

---

## 6. What was validated in this pass

- `tests/test_hpc_review_fixes.py` (+14) covers the SLURM poller, wait-timeout
  cancellation, job-id validation, `%N` throttling, checkpoint completeness
  gating, the triclinic box round-trip, the sin/cos embedding, the Amber
  cold-start fix, and the `device_index` sentinel.
- Full suite: **104 passed, 11 skipped** locally (skips need torch/MDAnalysis/
  deeptime, absent in the review sandbox); `ruff` clean; all 15 example configs +
  4 HPC configs validate against the updated schema.
- End-to-end scheduler behaviour must be validated on real clusters with
  `hpc_tests/` (SLURM/PBS × CPU/GPU).
