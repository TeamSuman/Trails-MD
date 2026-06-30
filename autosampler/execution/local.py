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


def _run_one(task: WalkerTask, device_index: int) -> bool:
    task.device_index = device_index
    # A single walker's failure (CUDA error, NaN blow-up, missing file, …) must
    # not abort the whole iteration — mirror the scheduler path (run_task.py),
    # which reports failures as success=False rather than raising.
    try:
        return run_walker_task(task)
    except Exception:  # noqa: BLE001 - report failure, keep the batch alive
        import logging
        import traceback

        logging.error(
            "Walker %s failed; marking it unsuccessful and continuing:\n%s",
            getattr(task, "index", "?"),
            traceback.format_exc(),
        )
        return False


class LocalProcessBackend(ExecutionBackend):
    def __init__(
        self, gpu_ids: list[int] | None = None, max_workers: int = 8, **_
    ):
        self.gpu_ids = gpu_ids
        self.max_workers = max_workers

    def execute(self, tasks: list[WalkerTask]) -> list[bool]:
        import multiprocessing as mp

        if not tasks:
            return []

        engine_name = tasks[0].engine_name
        engine_kwargs = tasks[0].engine_kwargs
        slots = _execution_slots(
            engine_name, engine_kwargs, self.gpu_ids, self.max_workers, len(tasks)
        )

        ctx = mp.get_context("spawn")
        results = [False] * len(tasks)
        task_iter = iter(tasks)

        def submit(executor, device_index: int):
            try:
                task = next(task_iter)
            except StopIteration:
                return None
            future = executor.submit(_run_one, task, device_index)
            return future, task.index, device_index

        with ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx) as executor:
            active: dict = {}
            for device_index in slots:
                submitted = submit(executor, device_index)
                if submitted is not None:
                    future, idx, dev = submitted
                    active[future] = (idx, dev)

            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, freed_device = active.pop(future)
                    try:
                        results[idx] = future.result()
                    except Exception:  # noqa: BLE001 - e.g. a killed worker process
                        import logging

                        logging.error(
                            "Walker %s did not return a result (worker died); "
                            "marking it unsuccessful.",
                            idx,
                        )
                        results[idx] = False
                    submitted = submit(executor, freed_device)
                    if submitted is not None:
                        next_future, next_idx, dev = submitted
                        active[next_future] = (next_idx, dev)

        return results


ExecutionBackendFactory.register("local", LocalProcessBackend)
