"""Local multi-process execution backend (multi-GPU workstation / single node).

Runs walkers as subprocesses across CPU worker slots or GPU device slots,
assigning GPU device indices dynamically as workers free up. This preserves the
original ``run_iteration_parallel`` behaviour behind the ExecutionBackend API.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from .base import ExecutionBackend, ExecutionBackendFactory, WalkerTask, run_walker_task


def _detect_gpu_ids() -> list[int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        try:
            return [int(device.strip()) for device in visible_devices.split(",")]
        except ValueError:
            pass
    try:
        import torch

        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0
    if num_gpus <= 0:
        return [0]
    return list(range(num_gpus))


def _uses_gpu_slots(engine_name: str, engine_kwargs: dict) -> bool:
    if engine_name == "amber":
        return "cuda" in str(engine_kwargs.get("amber_executable", "")).lower()
    if engine_name == "gromacs":
        return any(
            str(engine_kwargs.get(key, "")).lower() == "gpu"
            for key in (
                "gromacs_mdrun_nb",
                "gromacs_mdrun_pme",
                "gromacs_mdrun_update",
                "gromacs_mdrun_bonded",
            )
        )
    if engine_name == "openmm":
        platform = str(engine_kwargs.get("platform_name", "CUDA")).lower()
        return platform not in {"cpu", "reference"}
    return True


def _execution_slots(
    engine_name: str,
    engine_kwargs: dict,
    gpu_ids: list[int] | None,
    max_workers: int,
    n_walkers: int,
) -> list[int]:
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")
    if _uses_gpu_slots(engine_name, engine_kwargs):
        reserved = list(gpu_ids) if gpu_ids is not None else _detect_gpu_ids()
        if not reserved:
            reserved = [0]
        worker_count = min(max_workers, len(reserved), n_walkers)
        if worker_count <= 0:
            raise ValueError("max_workers must be greater than 0")
        return reserved[:worker_count]
    return [0 for _ in range(min(max_workers, n_walkers))]


# Per-worker-process cache of prepared, warm-reusable engines. Because each
# ProcessPoolExecutor worker is a distinct process that imports this module once,
# this dict is naturally private to a worker and persists across every task that
# worker handles — which is exactly the lifetime we want for a warm Context.
_WORKER_ENGINE_CACHE: dict = {}


def _run_one(task: WalkerTask, device_index: int, warm: bool = False) -> bool:
    task.device_index = device_index
    # A single walker's failure (CUDA error, NaN blow-up, missing file, …) must
    # not abort the whole iteration — mirror the scheduler path (run_task.py),
    # which reports failures as success=False rather than raising.
    try:
        cache = _WORKER_ENGINE_CACHE if warm else None
        return run_walker_task(task, engine_cache=cache)
    except Exception:  # noqa: BLE001 - report failure, keep the batch alive
        import logging
        import traceback

        logging.error(
            "Walker %s failed; marking it unsuccessful and continuing:\n%s",
            getattr(task, "index", "?"),
            traceback.format_exc(),
        )
        return False


def _terminate_workers(executor) -> None:
    """Best-effort kill of a ProcessPoolExecutor's worker processes."""
    for proc in list(getattr(executor, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001 - already gone / not terminable
            pass


# A single ProcessPoolExecutor kept alive across iterations for persistent_workers.
# It is module-level because ``make_backend`` builds a fresh backend object every
# iteration, so the pool cannot live on the backend instance. Keyed by worker
# count: if the slot count changes the old pool is retired and a new one built.
_PERSISTENT_POOL: dict = {"executor": None, "workers": None}


def _get_persistent_pool(n_workers: int, ctx):
    pool = _PERSISTENT_POOL
    if pool["executor"] is not None and pool["workers"] != n_workers:
        _retire_persistent_pool()
    if pool["executor"] is None:
        pool["executor"] = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        pool["workers"] = n_workers
    return pool["executor"]


def _retire_persistent_pool() -> None:
    executor = _PERSISTENT_POOL["executor"]
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    _PERSISTENT_POOL["executor"] = None
    _PERSISTENT_POOL["workers"] = None


class LocalProcessBackend(ExecutionBackend):
    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        max_workers: int = 8,
        walker_timeout: float | None = None,
        persistent_workers: bool = False,
        **_,
    ):
        self.gpu_ids = gpu_ids
        self.max_workers = max_workers
        self.walker_timeout = walker_timeout
        self.persistent_workers = persistent_workers

    def execute(self, tasks: list[WalkerTask]) -> list[bool]:
        import multiprocessing as mp
        import time

        if not tasks:
            return []

        engine_name = tasks[0].engine_name
        engine_kwargs = tasks[0].engine_kwargs
        slots = _execution_slots(
            engine_name, engine_kwargs, self.gpu_ids, self.max_workers, len(tasks)
        )

        ctx = mp.get_context("spawn")
        results = [False] * len(tasks)
        task_iter = iter(enumerate(tasks))
        timeout = self.walker_timeout
        warm = self.persistent_workers

        def submit(executor, device_index: int):
            try:
                pos, task = next(task_iter)
            except StopIteration:
                return None
            future = executor.submit(_run_one, task, device_index, warm)
            # value = (pos, task.index, device, start_time)
            return future, (pos, task.index, device_index, time.monotonic())

        def schedule(executor) -> None:
            active: dict = {}
            for device_index in slots:
                submitted = submit(executor, device_index)
                if submitted is not None:
                    future, meta = submitted
                    active[future] = meta

            while active:
                # Poll at the timeout cadence so overdue walkers are detected even
                # when nothing completes; `wait` itself returns no timed-out futures.
                poll = None if timeout is None else max(min(timeout, 30.0), 0.05)
                done, _ = wait(active, timeout=poll, return_when=FIRST_COMPLETED)
                for future in done:
                    pos, idx, freed_device, _ = active.pop(future)
                    try:
                        results[pos] = future.result()
                    except Exception:  # noqa: BLE001 - e.g. a killed worker process
                        import logging

                        logging.error(
                            "Walker %s did not return a result (worker died); "
                            "marking it unsuccessful.",
                            idx,
                        )
                        results[pos] = False
                    submitted = submit(executor, freed_device)
                    if submitted is not None:
                        next_future, meta = submitted
                        active[next_future] = meta

                if timeout is not None and active:
                    now = time.monotonic()
                    overdue = [m for m in active.values() if now - m[3] > timeout]
                    if overdue:
                        import logging

                        for pos, _idx, _dev, _start in active.values():
                            results[pos] = False
                        logging.error(
                            "Walker(s) %s exceeded walker_timeout=%ss; terminating "
                            "the batch and marking remaining walkers unsuccessful.",
                            sorted(m[1] for m in overdue),
                            timeout,
                        )
                        _terminate_workers(executor)
                        # A killed persistent pool holds dead workers (and lost warm
                        # contexts); retire it so the next iteration rebuilds cleanly.
                        if warm:
                            _retire_persistent_pool()
                        active.clear()

        if warm:
            schedule(_get_persistent_pool(len(slots), ctx))
        else:
            with ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx) as executor:
                schedule(executor)

        return results

    def close(self) -> None:
        """Retire the shared persistent pool, if any. Safe to call repeatedly."""
        if self.persistent_workers:
            _retire_persistent_pool()


ExecutionBackendFactory.register("local", LocalProcessBackend)
