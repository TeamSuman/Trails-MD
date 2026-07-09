# HPC scaling: how Trails-MD dispatches walkers, and how it compares to WESTPA

This page documents the execution model, its current limits at scale, and a
roadmap informed by how the mature weighted-ensemble package **WESTPA** handles
HPC job control.

## Trails-MD today: one array job per iteration

The scheduler backends (`trails_md/execution/{scheduler,slurm,pbs}.py`) dispatch
each adaptive iteration's walkers as **one scheduler array job**:

1. Pickle each `WalkerTask`, write a manifest line per task.
2. Render and submit a `#SBATCH --array=0-N` / `#PBS -J 0-N` script.
3. Poll `squeue`/`qstat`; collect per-walker **filesystem result markers**;
   resubmit missing/failed walkers up to `max_retries`.

Strengths: simple, portable, no long-lived daemons or open ports, and
unit-testable with a fake command-runner. Completion is filesystem-driven, so it
tolerates flaky queue accounting.

Costs at large fan-out:

- **Per-iteration submit/poll latency.** Every iteration pays a submit + queue +
  poll cycle. With short walkers this overhead can dominate.
- **Array-size ceilings.** SLURM `MaxArraySize` (default 1001) and PBS
  `max_array_size` cap the elements in a *single* array. Trails-MD splits a
  larger batch into sequential sub-arrays when you set `execution.max_array_size`
  (it does not auto-detect the site limit), so walkers-per-iteration can exceed
  the cap — at the cost of multiple submit/poll cycles per iteration.
- **Metadata pressure.** Thousands of small `*.pkl` / `result_*.json` / `.out` /
  `.err` files per campaign stress Lustre/GPFS metadata servers.
- **Coarse accounting.** Completion is inferred from marker files and a queue
  poll, not a task-level heartbeat, so a silently-dead worker is only noticed
  when its marker never appears.

## How WESTPA does it

WESTPA separates the *propagator* (the MD command) from a pluggable
**work manager** that distributes per-segment tasks. Managers include
`serial`, `threads`, `processes` (shared-memory), `mpi`, and — for multi-node —
`zmq` (a ZeroMQ master/worker pool). The recommended multi-node setup launches
**one allocation** whose SLURM/PBS script starts a WESTPA master (ZMQ server)
and a set of long-lived worker clients (via `srun`/`ssh` to a `node.sh`); workers
then **pull segment tasks over sockets for the entire run** (Unix sockets within
a node, TCP across nodes). GPU binding is computed per worker in `node.sh`
(a `CUDA_VISIBLE_DEVICES` per local rank), and the ZMQ manager uses worker
**heartbeats and task timeouts** to detect dead workers and resubmit their tasks.
Trajectory/weight/segment data lives in a single consolidated **HDF5** file
(`west.h5`) rather than many small files.

Sources:
[WESTPA work managers](https://github.com/westpa/westpa),
[Configuring WESTPA on SLURM](https://groups.google.com/g/westpa-users/c/xZU6LDfLblk),
[Multi-node ZMQ / multi-GPU](https://groups.google.com/g/westpa-users/c/18mts9s_rxI).

## Are we doing better or worse?

| Dimension | Trails-MD (array-per-iteration) | WESTPA (persistent ZMQ pool) |
| --- | --- | --- |
| Setup simplicity | **Better** — no daemons/ports, portable, testable off-cluster | Heavier (master + workers, TCP ports) |
| Per-iteration overhead | Worse — submit+poll every iteration | **Better** — pool amortized over whole run |
| Max walkers/iteration | Comparable — `max_array_size` chunking clears `MaxArraySize`, but each sub-array is a fresh submit | **Better** — pool size independent of batch |
| Dead-worker detection | Worse — inferred from missing markers | **Better** — heartbeats + task timeouts |
| Storage footprint | Worse — thousands of small files | **Better** — one HDF5 file |
| Locked-down clusters (no open ports) | **Better** — filesystem-only | Worse — needs socket connectivity |
| GPU binding | Comparable — inherits scheduler binding (after the fix) | Comparable — per-rank in `node.sh` |

Net: for **moderate** fan-out and portability, the array-job model is simpler and
robust. For **large, sustained** fan-out (hundreds–thousands of concurrent
walkers, many short iterations), WESTPA's persistent-pool model is materially
better on overhead, scale ceilings, and failure detection.

## Roadmap — ideas worth borrowing

1. **Persistent worker-pool backend (highest impact).** Add a work-manager-style
   backend that requests one allocation and keeps workers alive across
   iterations, streaming `WalkerTask`s to them. A dependency-free first cut can
   use an `mpi4py`/MPI backend (ubiquitous on HPC, no open ports) or a ZeroMQ
   backend behind the existing `ExecutionBackend` interface. This removes
   per-iteration submit latency and the array-size ceiling in one move.
2. **Auto-detect the array-size limit.** Array chunking
   (`execution.max_array_size`) and `%N` throttling (`execution.max_in_flight`)
   are implemented; what remains is querying the site limit automatically
   (`scontrol show config | grep MaxArraySize`, PBS `max_array_size`) so the user
   need not set `max_array_size` by hand.
3. **Task-level heartbeats/timeouts.** Even in the array model, have `run_task`
   emit periodic heartbeat files (or use `sacct`/`qstat -f` exit codes) so a
   hung or silently-killed walker is detected before `wait_timeout`.
4. **Consolidated HDF5 storage.** Replace the many-small-files layout
   (per-iteration trajectories/markers/pickles) with an HDF5/Zarr store to kill
   metadata-server pressure and make provenance a single artifact — closer to
   WESTPA's `west.h5`.
5. **Export the submit environment on PBS.** The manifest is already TAB-delimited
   (space-safe task/result paths, split with `cut -f`); what remains is making it
   easy to export the submit environment on PBS (e.g. `extra_directives: ["#PBS -V"]`).
6. **Torque flavor.** Add a `-t` / `PBS_ARRAYID` variant so classic Torque sites
   are supported alongside OpenPBS/PBS Pro.

See `hpc_tests/` for the validation suite that exercises the current backends and
that any of these changes should keep green.
