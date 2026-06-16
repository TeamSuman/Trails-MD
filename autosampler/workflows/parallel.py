from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Optional
import os

from autosampler.engines.amber import amber_trajectory_suffix

def _worker_task(engine_name: str, engine_kwargs: dict, prepare_kwargs: dict, run_kwargs: dict) -> bool:
    """Standalone function to instantiate engine in the worker process and run it."""
    import warnings
    warnings.filterwarnings("ignore", message="Non-optimal GB parameters detected for GB model HCT")
    warnings.filterwarnings("ignore", message="Reload offsets from trajectory")

    from autosampler.engines.base import EngineFactory

    engine = EngineFactory.get(engine_name, **engine_kwargs)
    engine.prepare(**prepare_kwargs)
    return engine.run_production(**run_kwargs)

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


def _reserved_gpu_slots(
    gpu_ids: Optional[list[int]], max_workers: int, n_walkers: int
) -> list[int]:
    reserved_gpu_ids = list(gpu_ids) if gpu_ids is not None else _detect_gpu_ids()
    if not reserved_gpu_ids:
        reserved_gpu_ids = [0]
    worker_count = min(max_workers, len(reserved_gpu_ids), n_walkers)
    if worker_count <= 0:
        raise ValueError("max_workers must be greater than 0")
    return reserved_gpu_ids[:worker_count]


def _uses_gpu_slots(
    engine_name: str,
    engine_kwargs: dict,
) -> bool:
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
    gpu_ids: Optional[list[int]],
    max_workers: int,
    n_walkers: int,
) -> list[int]:
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")
    if _uses_gpu_slots(engine_name, engine_kwargs):
        return _reserved_gpu_slots(gpu_ids, max_workers, n_walkers)
    return [0 for _ in range(min(max_workers, n_walkers))]


def run_iteration_parallel(engine_name: str, engine_kwargs: dict, prepare_kwargs: dict,
                           walkers: list, steps: int, stride: int,
                           outdir: Path, iteration: int, max_workers: int = 8,
                           gpu_ids: Optional[list[int]] = None) -> list:
    """Execute walker production runs over CPU worker slots or GPU device slots."""
    outdir.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp

    if not walkers:
        return []

    reserved_gpu_ids = _execution_slots(
        engine_name, engine_kwargs, gpu_ids, max_workers, len(walkers)
    )
    worker_count = len(reserved_gpu_ids)

    ctx = mp.get_context('spawn')
    results = [False] * len(walkers)
    walker_iter = iter(enumerate(walkers))

    def submit_walker(executor, gpu_id: int):
        try:
            idx, coords = next(walker_iter)
        except StopIteration:
            return None

        if engine_name == "amber":
            suffix = amber_trajectory_suffix(
                engine_kwargs.get("amber_trajectory_format", "auto"),
                engine_kwargs.get("amber_executable", "pmemd"),
            )
        else:
            suffix = "xtc"
        traj_out = outdir / f"iteration_{iteration}_{idx}.{suffix}"
        run_kwargs = {
            "run_index": idx,
            "start_coords": coords,
            "steps": steps,
            "traj_out": traj_out,
            "stride": stride,
            "device_index": gpu_id,
        }
        future = executor.submit(
            _worker_task,
            engine_name,
            engine_kwargs,
            prepare_kwargs,
            run_kwargs,
        )
        return future, idx, gpu_id

    with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
        active = {}
        for gpu_id in reserved_gpu_ids:
            submitted = submit_walker(executor, gpu_id)
            if submitted is not None:
                future, idx, assigned_gpu_id = submitted
                active[future] = (idx, assigned_gpu_id)

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                idx, freed_gpu_id = active.pop(future)
                results[idx] = future.result()
                submitted = submit_walker(executor, freed_gpu_id)
                if submitted is not None:
                    next_future, next_idx, assigned_gpu_id = submitted
                    active[next_future] = (next_idx, assigned_gpu_id)

    return results
