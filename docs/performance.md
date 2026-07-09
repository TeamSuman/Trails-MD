# Performance & GPU utilization

This page is the prioritized plan for running Trails-MD efficiently on HPC, with
an emphasis on **maximal GPU utilization**. It marks what is already implemented,
what is a quick win, and what is larger work, and points each item at the
`hpc_tests/` benchmark that measures it.

## Where the time goes

Trails-MD alternates short MD **bursts** with a thin **orchestration** step
(feature extraction → CV projection → spawn scoring → optional MSM → checkpoint).
For large systems MD dominates and the framework overhead is negligible (the
manuscript measures 4–7 % on a WW-domain multi-GPU run). For the *small* systems
used in tests and method development the opposite is true: **per-walker startup
(process spawn, imports, topology parse, GPU context creation, kernel JIT)** and
**analysis** can rival or exceed the integration time. The two levers that matter
most on HPC therefore are:

1. **Amortize startup** — do more MD per walker per iteration, and avoid paying
   process/import/JIT costs every iteration (persistent workers).
2. **Fill the GPU** — a small system uses a fraction of a modern GPU; run several
   walkers per card.

## 1. Maximal GPU utilization

### 1a. Pack multiple walkers per GPU (biggest single win for small systems)
A tiny protein occupies well under half of an A100/H100. Running one walker per
GPU wastes most of the card. Options, cheapest first:

- **CUDA MPS (Multi-Process Service).** Start the MPS control daemon on each GPU
  node and let several walker processes share one GPU concurrently, with the
  scheduler still handing each a `CUDA_VISIBLE_DEVICES`. Pack factor 2–8× for
  small systems. This needs no code change — it is a launch-time wrapper. Provide
  it via `execution.module_loads` / the SITE SETUP block (start `nvidia-cuda-mps-control -d`).
- **MIG (A100/H100).** Partition each physical GPU into MIG slices and expose one
  slice per walker for hard isolation *and* density. Configure at the site/queue
  level; each walker then sees a single MIG device.
- **Local backend, GPU slots.** On one big allocation (also the Torque fallback),
  set `spawning.max_workers` to `packing × n_gpus` and let the local backend's GPU
  slot assignment spread walkers; combine with MPS for concurrency on a slot.

*Benchmark:* run `hpc_tests/slurm/run_gpu.sbatch` with and without MPS at fixed
walker count and compare `Runner` time in `output.log`; confirm isolation with the
`GPU_BINDING` check (`<traj>.gpu.json` markers).

### 1b. Longer walkers per iteration
Increase `spawning.step` (and reduce iteration count for the same budget) so
integration dominates startup. This is the simplest knob and often the largest
gain for method-development workloads. Watch the adaptive trade-off: longer bursts
decorrelate more but adapt less frequently.

### 1c. Persistent, resident GPU workers (larger effort, biggest scaling win)
Today each walker is a fresh process (scheduler array element or local subprocess)
that re-imports, re-parses topology, and re-creates the GPU context every
iteration. A **persistent worker pool** — long-lived processes that hold a warm
OpenMM `Context`/`System` and receive `(positions, velocities, steps)` tasks over
a queue — eliminates per-iteration startup and JIT, and keeps the GPU busy across
iterations. This is the model mature tools (WESTPA) use and is already on the
roadmap in [HPC scaling](hpc_scaling.md). Highest value at high walker counts and
short bursts.

### 1d. Engine GPU-resident settings
- **OpenMM:** `precision: single` is materially faster than `mixed`/`double` for
  small systems where accuracy allows; keep PME on the GPU (do not set
  `UseCpuPme`). Device isolation (CUDA/OpenCL/HIP) and CPU thread caps are
  **implemented** (`engines/openmm.py::_platform_properties`).
- **GROMACS:** run GPU-resident — `-nb gpu -pme gpu -bonded gpu -update gpu`
  with `-ntmpi 1 -pin on` and a per-walker `-pinoffset`. These are already
  configurable (`engine.gromacs_mdrun_{nb,pme,bonded,update,pin}`,
  `gromacs_mdrun_ntomp`); `-ntomp` is derived from the scheduler CPU allocation
  (`engines/gromacs.py::_resolve_ntomp`). A future quick win is auto-deriving
  `-pinoffset` per local GPU/CPU slot.
- **Amber:** use `pmemd.cuda`, one walker per GPU (or per MIG slice); NetCDF output.

## 2. Scheduler & throughput

- **Array chunking + concurrency cap** — `execution.max_array_size` (split large
  batches into sub-arrays) and `execution.max_in_flight` (SLURM `%N`) are
  **implemented**; use them to stay under `MaxArraySize` and site rate limits.
- **GPU binding validation** — **implemented** (`GPU_BINDING` check) so a silent
  CUDA→CPU fallback or all-walkers-on-device-0 is caught automatically.
- **Job packing (larger effort)** — for very many tiny walkers, run several
  walkers per array element to cut scheduler transaction overhead (complements
  MPS). Pairs naturally with the persistent-worker model.

## 3. Filesystem / inode pressure

Per walker per iteration Trails-MD writes a trajectory, and per iteration a
`cvs.npz`, checkpoint delta, and (optionally) `msm.npz`. Over a long campaign on a
shared Lustre/GPFS filesystem this is heavy metadata traffic.

- **Consolidate (larger effort):** bundle an iteration's per-walker trajectories +
  metadata into a single HDF5/Zarr archive or per-iteration tar, cutting inode/stat
  load by the walker count.
- **Stage through node-local scratch:** write walker outputs to node-local
  `$TMPDIR` and copy the consolidated result back once per iteration.
- **Reduce per-iteration stat load (quick win, implemented-adjacent):** the history
  usability prune re-stats the whole cumulative trajectory set each iteration
  (`core.py::_prune_unusable_history_trajectories`), which is O(iterations × walkers)
  metadata calls per iteration. Restrict it to the current iteration's new files
  plus an on-resume full sweep.

## 4. Analysis pipeline

- **Overlap analysis with MD (larger effort):** project CVs / estimate the MSM for
  iteration *k* while iteration *k+1*'s MD runs, hiding orchestration behind the GPU.
- **Incremental estimators (medium effort):** update TICA covariances and the MSM
  count matrix incrementally instead of refitting from the full history each retrain
  / cadence; reuse clustering across iterations (`msm.stable_clustering` already
  supports comparable microstates).
- **Bounded feature memory (implemented):** `max_adaptive_memory_frames` caps the
  reprojected feature history so retraining cost does not grow without bound.
- **Parallelize the implied-timescale sweep (quick win):** the `lagtimes` diagnostic
  sweep is embarrassingly parallel.

## 5. Reproducibility vs. speed

Deterministic per-walker seeds are threaded to all engines, and GROMACS `ld-seed`
is pinned. Bitwise GPU reproducibility (OpenMM `DeterministicForces`, CUDA
non-determinism) costs performance and is off by default; enable only when exact
reproduction is required.

## Priority summary

| Item | Effort | Status | Benchmark |
| --- | --- | --- | --- |
| Longer walkers per iteration (1b) | trivial | knob | any GPU run: `Runner` time vs `step` |
| Multiple walkers per GPU via MPS (1a) | low (site) | site config | `run_gpu.sbatch` ± MPS + `GPU_BINDING` |
| MIG per-walker slices (1a) | low (site) | site config | `GPU_BINDING` device spread |
| GROMACS GPU-resident flags (1d) | low | configurable | `gromacs_*` runs: `Runner` time |
| OpenMM `single` precision (1d) | trivial | configurable | latent-CV runs: `Runner` time |
| Parallel implied-timescale sweep (4) | low | todo | `openmm_tica_msm` wall time |
| Reduce per-iteration stat load (3) | low | todo | long campaign iteration `Other` time |
| Incremental TICA/MSM (4) | medium | todo | `openmm_msm_convergence` `Other` time |
| Consolidate trajectories to HDF5/Zarr (3) | medium | todo | inode count / campaign |
| Persistent resident worker pool (1c) | high | roadmap | high-walker-count scaling |
| Overlapped analysis pipeline (4) | high | todo | `Other`/`Runner` overlap |

See [HPC scaling](hpc_scaling.md) for the dispatch model and the persistent-worker
roadmap, and `hpc_tests/RUNBOOK.md` for how to run the benchmarks above.
