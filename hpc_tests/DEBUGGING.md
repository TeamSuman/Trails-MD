# Trails-MD HPC debugging playbook

Audience: a person — or a Claude Code agent — triaging a failed cluster run.
Start from the structured JSON reports, then use the sections below, which are
keyed to the failure `code`s emitted by `checks/validate_results.py` and the
checks in `checks/preflight.py`. Each entry gives **symptom → likely cause →
where to look in the code → fix/action**.

## How to triage (do this first)

1. Read `results/<scheduler>_<queue>/preflight.json` → `overall`. If it is
   `fail`, fix the environment before looking at anything else (a required
   check failed: package import, scheduler tools, or shared filesystem).
2. Read `results/<scheduler>_<queue>/validate.json` → `overall` and the list of
   `FAILED codes`.
3. Read `run.log` (the orchestrator's stdout/stderr) and the per-walker logs
   under `<outdir>/iter_*/_jobs/logs_attempt*/` and result markers under
   `<outdir>/iter_*/_jobs/result_*.json` (each has `error` + `traceback` on
   failure — this is the ground truth for *why a walker failed*).
4. Look up each failing code below.

Key source files: `trails_md/execution/scheduler.py` (submit/poll/retry),
`trails_md/execution/{slurm,pbs}.py` (directives + polling),
`trails_md/execution/run_task.py` (per-walker entry point + result marker),
`trails_md/engines/{openmm,gromacs,amber}.py` (the MD itself).

---

## Preflight codes

### `trails_md_import = fail`
The package is not importable in the job environment. Cause: `module_loads` /
`conda activate` in the driver's SITE SETUP block (and in the config's
`execution.module_loads`) do not put `trails-md` on `PYTHONPATH`. Fix: activate
the same env you installed into; verify with `python -c "import trails_md"`.

### `slurm_tools_on_path` / `pbs_tools_on_path = fail`
`sbatch/squeue/scancel` (or `qsub/qstat/qdel`) not found. You are on a node
without scheduler client tools, or a module must be loaded. Fix: submit from a
login/submit node; `module load slurm` if your site requires it.

### `filesystem = fail`
The working directory is not writable, or (worse) it is node-local. The
scheduler backend **requires a shared filesystem** visible to both the submit
host and every compute node, because completion is detected via result-marker
files (`scheduler.py:_read_success`). Fix: run from a shared scratch/project
path (Lustre/GPFS/NFS), never node-local `/tmp`.

### `config_validates = fail`
The YAML fails schema validation (`trails_md/config.py`). The `detail` string is
the pydantic error. Common: bad `walltime`/`memory` format, negative resource,
`min_success_fraction` out of `(0,1]`.

---

## Result / run codes

### `OUTDIR_EXISTS`
The run output directory named on the `--outdir` you passed to
`validate_results.py` does not exist. Either the run never started (read the tail
of `run.log` — a `--check`/import failure aborts before any output), or the
`--outdir` does not match the config's `outdir` (paths in the config are resolved
relative to the config file). All other checks are skipped until this passes.

### `RESULT_MARKERS` (0 successful markers, scheduler backend)  ← highest-signal
**Symptom:** the run reports every walker failed almost immediately, yet
`squeue`/`qstat` showed the array actually running; trajectories may even exist.
**Cause (historical bug, now fixed):** the SLURM poller's "is the job still
active?" check did not recognize `squeue --array` output (`12345_0`), so the
driver believed the job had left the queue and gave up after a short grace,
before any walker finished. See `trails_md/execution/slurm.py:_job_active`
(now matches `^\s*<jobid>(?:_|\b)` line-anchored). **If this recurs**, print what
`squeue --job <id> --noheader --array` actually returns on your site and confirm
`_job_active` matches it; some sites customize the JOBID format. As a fallback,
completion is *also* driven by result-marker files, so raise
`execution.marker_grace` (seconds to keep re-checking markers after the job
leaves the queue) if markers are merely slow to appear on the shared FS.

### `TRAJ_FILES` (missing / empty per-walker trajectories)
**Symptom:** expected `iteration_<it>_<w>.xtc` (or `.nc`) missing or zero-byte.
**Diagnose:** open the matching `result_<w>.json` (has `error`+`traceback`) and
the array element's `.err` log. Common causes:
- Engine env not reconstructed in the array job (module/conda not loaded) →
  `openmm`/`gmx`/`pmemd` import/exec fails. Fix `execution.module_loads`.
- OpenMM CUDA platform requested on a node that got no GPU → see `GPU_BINDING`.
- MD blew up (NaN) from a bad start frame → check the engine `.err`.
- Disk full / quota on the shared FS.

### `WALKER_FAILURES` (non-zero failed walkers in output.log)
Informational unless `min_success_fraction < 1.0`. With the default `1.0`, any
failed walker aborts the iteration (`core.py: run_iteration`). To let a long
campaign shrug off transient failures, set e.g. `min_success_fraction: 0.9`;
failed walkers are then dropped and sampling continues with the survivors.

### `ITER_DIRS` / `LOG_ITERATIONS` (fewer iterations than expected)
The run stopped early. Read the *tail* of `run.log`:
- `RuntimeError: N/M walker(s) failed ... (min_success_fraction=...)` → walkers
  failed; go to `TRAJ_FILES`.
- `submission failed (exit ...)` → the array `sbatch`/`qsub` was rejected; see
  `ARRAY_LIMIT` and `PBS_FLAVOR`.
- `Converged: ...` → not a failure; the campaign converged early.
- The driver hit its own `#SBATCH --time` / `#PBS walltime` → raise it.

### `CHECKPOINTS` (missing or incomplete)
- No `checkpoints/` dir → `checkpoint_freq: 0` disables checkpointing (the run
  logs a loud warning at startup). Set `checkpoint_freq >= 1` for restartability.
- `incomplete_no_marker` non-empty → a checkpoint dir lacks its `format_version`
  completion marker, i.e. the job died mid-save. This is expected after a hard
  kill; `latest_iteration()` skips it and resume falls back to the last complete
  checkpoint (`trails_md/checkpoints/manager.py`). No action unless *every*
  checkpoint is incomplete (then the FS lost writes — investigate the shared FS).

---

## Feature-test codes (opt-in checks from `run_local_matrix.py` / `--check-*`)

These are emitted only when the corresponding feature check is requested (the
feature matrix requests them automatically). See [`RUNBOOK.md`](RUNBOOK.md) §1.

### `LATENT_DIM` (learned-CV runs)
**Symptom:** the last `iter_*/cvs.npz` has a column count ≠ the configured
`adaptive_model.latent_dim`. **Cause:** the CV model trained to the wrong
dimension, or projection fell back to physical CVs. **Look at:**
`trails_md/spaces/model.py` (`fit`/`project`) and the run's `space_mode`; confirm
the learned backend is installed (`space_mode` needs `deeptime`/`torch`).

### `MSM_NPZ` (in-loop MSM runs)
**Symptom:** no `iter_*/msm.npz`, or its `timescales` are non-finite / its
`transition_matrix` rows do not sum to 1. **Cause:** MSM estimation was skipped
(cumulative frames `< msm.min_frames`), or the connected set had `< 2` states
(sampling too sparse / lagtime too large). **Look at:** `run.log` for
`MSM skipped …`/`MSM estimation failed …`, and `trails_md/msm/estimator.py`.
Lower `msm.min_frames`/`msm.lagtime`/`msm.n_microstates` for a small smoke test.
Note the in-loop MSM is **experimental** (see `docs/msm.md`).

### `RESUME_CHAIN` (resume runs)
**Symptom:** the delta-checkpoint chain does not reconstruct to a gapless history.
**Cause:** a per-iteration `history.pkl` delta was lost/torn, so
`reconstruct_history` reports a broken chain. **Look at:**
`trails_md/checkpoints/manager.py` (`reconstruct_history`, `history_chain.json`)
and whether the shared FS dropped a write. Not a resume *code* bug on its own —
usually a missing/partial checkpoint dir.

### `PATH_OUTPUT` (path reconstruction)
**Symptom:** `trails-md-path` produced no or a zero-byte output trajectory.
**Cause:** the requested start/end CVs share no recorded lineage, or MDAnalysis /
`gmx` could not write frames. **Look at:** `trails_md/paths.py`
(`connected_record_path`, `write_connected_trajectory`); confirm the endpoints
fall within the sampled region and that MDAnalysis is installed (else set
`TRAILS_MD_GMX`).

### `GPU_BINDING` (per-walker GPU device isolation)
Requested with `--check-gpu-binding` (the GPU feature runs and the GPU smoke
scripts add it automatically). The OpenMM engine writes a `<trajectory>.gpu.json`
marker per walker recording the *resolved* platform and device
(`engines/openmm.py::_write_gpu_binding_marker`); the check reads them.
**Fails when:**
- **Any walker ran on CPU** while a GPU platform was requested — a bad device pin
  silently degraded to CPU (`engines/openmm.py::_create_simulation` fallback).
  Fix the device visibility / driver so the pin succeeds.
- **Walkers piled onto one device** (`--gpu-count > 1` given, fewer distinct
  devices used than `min(walkers, gpu-count)`) — the scheduler did not isolate a
  GPU per task. On SLURM use `--gpus-per-task=1` **with** cgroup
  `ConstrainDevices=yes`; on PBS many sites do not cgroup-isolate GPUs, so request
  whole nodes or set `CUDA_VISIBLE_DEVICES` from the local rank in
  `execution.module_loads`. Verify by reading `$OUTDIR/iter_*/*.gpu.json` (the raw
  device each walker used) or the per-walker `logs_attempt*/*.out`.

Without `--gpu-count` the check is report-only (it cannot tell a correct
single-GPU node from missing isolation) but still fails on a CPU fallback and
prints the device distribution.

---

## Cross-cutting HPC symptoms (not tied to one code)

### The whole driver hangs for a long time, no progress
A walker array job is stuck in the queue (held `H`, or unschedulable resource
request) and never runs, so markers never appear. The poller now enforces an
overall `wait_timeout` (`scheduler.py:_wait_for_completion`, derived from
`walltime` when unset) and will `scancel`/`qdel` and mark the batch failed.
If it still hangs: your `walltime` is huge (so the derived ceiling is huge) —
set `execution.wait_timeout` explicitly, and check `squeue`/`qstat` for the
array's state and the reason (`--start`, `-f`).

### Submission rejected above ~1000 walkers (SLURM)
`code`: `ARRAY_LIMIT`. SLURM `MaxArraySize` defaults to 1001 (max index 1000);
PBS has a site `max_array_size`. This caps a *single* array. Trails-MD splits a
larger batch into sequential sub-arrays when you set `execution.max_array_size`
below the site limit (`scheduler.py:_dispatch_attempt`), so an iteration with
more walkers than the cap still submits — as several arrays. If you hit a
rejection, either the batch exceeds the cap and `max_array_size` is unset/too
high, or the value you set still exceeds the site limit. Check the limit with
`scontrol show config | grep -i MaxArraySize` and set `execution.max_array_size`
under it; also use `execution.max_in_flight` (`%N`) to throttle concurrency. For
very large sustained fan-out see the persistent-worker-pool roadmap item in
`docs/hpc_scaling.md`.

### GPU contention / all walkers on device 0 (`GPU_BINDING`)
This is now validated automatically — see the `GPU_BINDING` feature-test code
above for the `<traj>.gpu.json` markers and the pass/fail rules. The background
below explains *why* isolation can fail and how to fix it at the site.
**Symptom:** GPU test runs, but `nvidia-smi` shows all walkers on one GPU while
others idle (throughput ~1/Nth of expected).
**Root behaviour:** on the scheduler path each walker inherits the GPU the
scheduler bound via `CUDA_VISIBLE_DEVICES` — the engines no longer force
`CUDA_VISIBLE_DEVICES=0` / `DeviceIndex=0` (the `WalkerTask.device_index = -1`
sentinel means "external binding"). For this to isolate GPUs, your site must
constrain devices per job:
- SLURM: `--gpus-per-task=1` **with** cgroup device isolation
  (`ConstrainDevices=yes` in cgroup.conf). Verify by printing
  `echo "$CUDA_VISIBLE_DEVICES"` from a walker (add it to `module_loads` as an
  `echo` for one run, or read the element `.out`).
- PBS: many sites do **not** cgroup-isolate GPUs. If `ngpus=1` does not set
  `CUDA_VISIBLE_DEVICES`, walkers co-located on a node will all see every GPU.
  Workaround: request whole nodes, or set `CUDA_VISIBLE_DEVICES` from the local
  rank in `execution.module_loads`.
Confirm which GPU each walker used from `logs_attempt*/*.out` (OpenMM logs the
CUDA fallback message on a device error; a bad `DeviceIndex` now degrades to CPU
instead of crashing — see `engines/openmm.py:_create_simulation`).

### PBS array job rejected or `PBS_ARRAY_INDEX: unbound variable` (`PBS_FLAVOR`)
The PBS backend targets **OpenPBS / PBS Pro** (`#PBS -J`, `PBS_ARRAY_INDEX`).
Classic **Torque** uses `#PBS -t` and `PBS_ARRAYID` and will fail. `preflight.py
--scheduler pbs` now auto-detects this (the `pbs_flavor` check warns on Torque),
so run preflight first. If you are on Torque, this backend needs a Torque variant
(see roadmap); as a stopgap, use the **`local` backend within a single large
multi-node/GPU allocation** — set `execution.backend: local` and
`spawning.max_workers` to the number of GPU/CPU slots in the allocation.

### Poll commands time out under load
At scale, `squeue`/`qstat` can exceed `execution.submit_timeout`. The poller now
catches the timeout and retries on the next cycle instead of crashing
(`scheduler.py:_wait_for_completion`). If it is chronic, raise `submit_timeout`
and `poll_interval` to reduce scheduler query pressure.

### Manifest / path parsing breaks
The array-job manifest is **TAB-delimited** and split with `cut -f1`/`cut -f2`
(`scheduler.py:_render_script`), so task/result paths containing spaces survive.
If walkers still fail on path parsing, confirm the rendered `submit_*.sh` under
`<outdir>/iter_*/_jobs/` uses `cut -f` (not `cut -d" "`) and that a literal TAB
separates the two columns of `manifest_*.txt`; a shell that rewrites TABs (rare)
or an edited script could reintroduce the problem. A space-free `outdir` remains
the safest choice regardless.

---

## Reporting back (for an automated agent)

When you have a diagnosis, produce:
1. The failing `code`(s) and the single root cause.
2. The exact file:line in `trails_md/` implicated (from the sections above).
3. Whether it is a **site/config** issue (fix the YAML/driver) or a **code** bug
   (propose a patch + a regression test mirroring `tests/test_execution.py`,
   which uses a fake command-runner so scheduler logic is testable off-cluster).
4. The relevant excerpt from `run.log` / a `result_*.json` `traceback`.
