# Trails-MD HPC test suite

End-to-end validation of the **SLURM** and **PBS** execution backends on real
clusters, across **CPU-only** and **GPU** queues. These tests exist because the
scheduler backends cannot be fully exercised off-cluster (the unit tests in
`tests/test_execution.py` use a synchronous fake scheduler); only a live
`sbatch`/`qsub` + `squeue`/`qstat` loop surfaces polling, GPU-binding,
filesystem-lag, and resource-request problems.

Workflow for each test: **preflight → `trails-md --check` → run → validate**,
with structured JSON written to `results/` so a person *or an automated agent*
can localize any failure using [`DEBUGGING.md`](DEBUGGING.md).

## Test matrix

| Scheduler | Queue | Driver script | Config |
| --- | --- | --- | --- |
| SLURM | CPU-only | `slurm/run_cpu.sbatch` | `configs/alad_cpu_slurm.yaml` |
| SLURM | GPU | `slurm/run_gpu.sbatch` | `configs/alad_gpu_slurm.yaml` |
| PBS   | CPU-only | `pbs/run_cpu.pbs` | `configs/alad_cpu_pbs.yaml` |
| PBS   | GPU | `pbs/run_gpu.pbs` | `configs/alad_gpu_pbs.yaml` |

All four use the **same tiny alanine-dipeptide workload** (vacuum Amber14 system,
no external force fields, 8 walkers × 2–10 ps). It runs in a few minutes, so the
matrix isolates *scheduler/HPC behaviour* from force-field/asset complexity. The
CPU and GPU variants differ only in `engine.platform_name` (CPU vs CUDA) and the
per-task resource request.

## Prerequisites (once per cluster)

```bash
# 1. Clone and create the environment on a SHARED filesystem (not node-local).
git clone <this-repo> && cd Trails-MD
conda env create -f env.yml           # or: conda create -n trails-md python=3.11
conda activate trails-md
pip install -e '.[openmm]'            # OpenMM from conda-forge is most reliable
trails-md --help                      # sanity check the console script resolves
```

## Configure for your site

Edit **two** places consistently:

1. The **driver script** `SITE SETUP` block (module loads + `conda activate`).
2. The **config** `execution` block: `partition`, `account`, `module_loads`
   (must reproduce the same environment for the per-walker array jobs),
   `extra_directives`, and `memory`. The per-walker jobs run on *fresh* shells,
   so `module_loads`/`extra_directives` must fully reconstruct the runtime
   environment (on PBS, consider `extra_directives: ["#PBS -V"]`).

## Run

From the repo root:

```bash
# SLURM
sbatch hpc_tests/slurm/run_cpu.sbatch
sbatch hpc_tests/slurm/run_gpu.sbatch

# PBS
qsub hpc_tests/pbs/run_cpu.pbs
qsub hpc_tests/pbs/run_gpu.pbs
```

Override defaults via environment variables, e.g. a scaling test:

```bash
# SLURM: 32 walkers, 5 iterations
CONFIG=hpc_tests/configs/alad_cpu_slurm.yaml ITERATIONS=5 WALKERS=32 \
  sbatch --export=ALL hpc_tests/slurm/run_cpu.sbatch
```
(Also raise `spawning.walker` to 32 and `execution.max_in_flight` in the config.)

## What each test validates

- **Submission & array sizing** — `sbatch --parsable` / `qsub` succeed; the array
  directive is well-formed (`--array=0-N%M` / `#PBS -J 0-N`).
- **Polling correctness** — the driver actually *waits* for walkers to finish
  (this is the regression guard for the SLURM `squeue --array` poller bug: it
  must not declare the job done while elements are still running).
- **Completion via filesystem markers** — every walker writes `result_*.json`;
  every expected trajectory exists and is non-empty.
- **GPU binding** (GPU tests) — each walker uses the scheduler-allocated GPU
  rather than all piling onto device 0.
- **Checkpointing** — checkpoints are written and carry the `format_version`
  completion marker (torn checkpoints are detectable/skippable).
- **Robustness knobs** — `max_retries`, `max_in_flight`, `marker_grace`, and a
  derived `wait_timeout` behave as configured.

## Interpreting results

Each run writes `results/<scheduler>_<queue>/{preflight,validate}.json` and
`run.log`. Read `validate.json` → `overall` first. On any failure, open
[`DEBUGGING.md`](DEBUGGING.md) and look up the failing `code`
(e.g. `RESULT_MARKERS`, `GPU_BINDING`, `TRAJ_FILES`): each entry gives the
likely cause, where in the code to look, and how to fix or work around it.

## Scaling ladder (recommended once the smoke tests pass)

Run the CPU SLURM/PBS test at `WALKERS = 8, 64, 256, 1024` and record wall-clock
per iteration and scheduler behaviour. Watch for: array-size limits
(`MaxArraySize`, PBS `max_array_size`), submit-rate throttling, `_jobs/`
small-file pressure on the shared FS, and poll-command latency. Note what breaks
and at what N — this directly informs whether to adopt the persistent-worker-pool
model discussed in `docs/hpc_scaling.md` (WESTPA-style) for very large fan-out.
