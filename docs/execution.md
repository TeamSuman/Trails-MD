# Execution: workstation & HPC

Where walkers run is a pure configuration choice via the `execution` section.
The science is identical across backends.

## Local (multi-GPU workstation)

The default. Walkers run as subprocesses across CPU worker slots or GPU device
slots, with GPU device ids assigned dynamically as workers free up.

```yaml
execution:
  backend: local
engine:
  platform_name: CUDA
  gpu_ids: [0, 1, 2, 3]     # optional; auto-detected otherwise
spawning:
  max_workers: 4            # concurrent walkers
```

Each walker is pinned to a specific GPU slot: OpenMM uses the CUDA
`DeviceIndex` platform property, while the GROMACS and Amber subprocess engines
set `CUDA_VISIBLE_DEVICES`. With more walkers than slots, walkers are streamed
onto slots as they free up.

!!! note "Restricted GPU visibility"
    If you pre-set `CUDA_VISIBLE_DEVICES` to a non-contiguous subset before
    launching the local backend, OpenMM's `DeviceIndex` is numbered *relative*
    to the visible set. Prefer one worker per fully-visible GPU, or let a
    scheduler backend do the binding (below).

## SLURM

Each iteration's walkers are submitted as **one array job**
(`#SBATCH --array=0-N`). Trails-MD renders the script, submits with
`sbatch --parsable`, polls `squeue`, and collects per-walker result markers from
the shared filesystem. Failed or missing walkers are resubmitted up to
`max_retries` times.

```yaml
execution:
  backend: slurm
  partition: gpu
  account: my_alloc
  walltime: "02:00:00"
  cpus_per_task: 8
  gpus_per_task: 1          # some sites reject this; use `gres` instead (below)
  memory: "16G"
  max_retries: 2
  poll_interval: 30
  max_in_flight: 64         # cap concurrent array elements (SLURM `%N`)
  marker_grace: 30          # tolerate shared-FS metadata lag (seconds)
  wait_timeout: null        # ceiling on waiting for one array job; null = derive from walltime
  module_loads:
    - "module load cuda/12.2"
    - "module load openmm"
  extra_directives:
    - "#SBATCH --qos=normal"
```

### Robustness at scale

- `max_in_flight` caps how many array elements run at once (rendered as
  `--array=0-N%max_in_flight`), so a batch of hundreds/thousands of walkers does
  not flood the scheduler or hit submit-rate limits.
- `wait_timeout` bounds how long the driver waits for one iteration's array job
  before cancelling it (`scancel`/`qdel`) and treating unfinished walkers as
  failed — a held or unschedulable job cannot hang the campaign. Left `null`, a
  generous ceiling is derived from `walltime`.
- `marker_grace` keeps re-checking result markers after the job leaves the queue,
  absorbing NFS/Lustre/GPFS metadata lag so genuinely-successful walkers are not
  misreported as failed.

!!! note "Array-size limits"
    One array element is submitted per walker. SLURM's `MaxArraySize` (default
    1001) and PBS's `max_array_size` cap a single array; set `max_array_size` to
    split a larger batch into sequential sub-arrays. For very large fan-out see
    [HPC scaling](hpc_scaling.md).

## PBS (OpenPBS / PBS Pro)

The same model with PBS array jobs (`#PBS -J 0-N`, `qsub`, `qstat`). This backend
targets **OpenPBS / PBS Pro** (`#PBS -J`, `PBS_ARRAY_INDEX`); classic **Torque**
(`#PBS -t`, `PBS_ARRAYID`) is **not** currently supported. On PBS the submit
environment is not exported by default — reconstruct it via `module_loads`, or
add `"#PBS -V"` to `extra_directives`:

```yaml
execution:
  backend: pbs
  partition: gpuq           # PBS queue
  walltime: "02:00:00"
  cpus_per_task: 8
  gpus_per_task: 1
  memory: "16gb"
  module_loads:
    - "module load openmm"
```

## How it works

- Each walker is a self-contained task pickled to the iteration's `_jobs/`
  directory. An array element loads its task and runs
  `python -m trails_md.execution.run_task`, writing a JSON **result marker**.
- **Completion is filesystem-driven** (result markers), not scheduler
  accounting — robust to flaky queue state. A walker that dies without a marker
  is treated as failed and resubmitted.
- Requirements: a **shared filesystem** visible to compute nodes, and the
  `trails-md` package importable in the job environment (hence `module_loads`
  / activating your conda env in the job, e.g. via `extra_directives`).

## Choosing resources

`cpus_per_task` / `gpus_per_task` / `memory` are **per walker** (one array
element). For CPU-only HPC, set `gpus_per_task: 0` and an OpenMM `CPU`
platform (or a CPU GROMACS/Amber build) and scale out across many array tasks.

## Choosing your MD engine

The engine is `engine.md_engine` (`openmm` | `gromacs` | `amber`). **OpenMM** is
used natively through its Python API — no external executable, just the `openmm`
package importable in the job environment. **GROMACS** and **Amber** are external
programs, so you select them by making the executable available (module or path)
and pointing the config at it:

```yaml
engine:
  md_engine: gromacs
  gromacs_executable: gmx            # a name on PATH, or an absolute path
  gromacs_include_dir: /opt/gromacs/share/gromacs/top   # holds the *.ff force-field dirs
```

```yaml
engine:
  md_engine: amber
  amber_executable: pmemd.cuda       # pmemd | pmemd.cuda | sander; PATH or absolute path
```

On a scheduler backend the array jobs run in **fresh shells**, so the engine's
module must be loaded there too: add it to `execution.module_loads` (e.g.
`"module load gromacs/2024"`), which is replayed inside every walker job. Giving
an **absolute** `*_executable` path avoids depending on a module to set `PATH`.
The HPC test suite exercises all three engines this way — see
[`hpc_tests/RUNBOOK.md`](https://github.com/TeamSuman/Trails-MD/blob/main/hpc_tests/RUNBOOK.md).

## Fault tolerance for long campaigns

By default (`min_success_fraction: 1.0`) any walker failure aborts the iteration
so nothing proceeds on partial data. For long multi-day HPC campaigns where an
occasional transient node/GPU/integrator failure is expected, set a threshold
below 1.0:

```yaml
min_success_fraction: 0.9   # continue if >= 90% of walkers succeed
```

Failed walkers (and their lineage parents) are dropped and sampling continues
with the survivors; the spawner restores the full walker count next iteration.
Scheduler backends additionally resubmit failed walkers up to `max_retries`
times before they count as failures.

## Testing on your cluster

Before a production campaign, validate the backend end-to-end with the
ready-made suite in [`hpc_tests/`](https://github.com/TeamSuman/Trails-MD/tree/main/hpc_tests)
(SLURM + PBS, CPU-only + GPU). It runs preflight → `--check` → a tiny run →
result validation and writes structured JSON, with a debugging playbook
(`hpc_tests/DEBUGGING.md`) keyed to each failure mode.

## Adding a scheduler

Subclass `SchedulerBackend` (`trails_md/execution/scheduler.py`), implement
the directive/submit/poll hooks, and call
`ExecutionBackendFactory.register(...)`. The submit → poll → collect → retry
machinery is inherited. For the persistent-worker-pool model used by mature
tools like WESTPA, see [HPC scaling](hpc_scaling.md).
