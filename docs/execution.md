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

GPU visibility per worker is set through `CUDA_VISIBLE_DEVICES`. With more
walkers than slots, walkers are streamed onto slots as they free up.

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
  gpus_per_task: 1
  memory: "16G"
  max_retries: 2
  poll_interval: 30
  module_loads:
    - "module load cuda/12.2"
    - "module load openmm"
  extra_directives:
    - "#SBATCH --qos=normal"
```

## PBS / Torque (PBS Pro)

The same model with PBS array jobs (`#PBS -J 0-N`, `qsub`, `qstat`):

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

## Adding a scheduler

Subclass `SchedulerBackend` (`trails_md/execution/scheduler.py`), implement
the directive/submit/poll hooks, and call
`ExecutionBackendFactory.register(...)`. The submit → poll → collect → retry
machinery is inherited.
