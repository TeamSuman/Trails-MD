"""Pluggable execution backends for dispatching walker MD jobs.

A backend turns a list of :class:`WalkerTask` (one short MD run each) into a
list of per-walker success flags. Implementations:

- ``local``  — subprocesses across CPU/GPU slots on one node (workstation).
- ``slurm`` / ``pbs`` — one scheduler array job per iteration (HPC clusters).

All backends share the same ``execute(tasks) -> list[bool]`` contract so the
orchestrator is agnostic to where walkers actually run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trails_md.engines.amber import amber_trajectory_suffix


def _traj_suffix(engine_name: str, engine_kwargs: dict) -> str:
    if engine_name == "amber":
        return amber_trajectory_suffix(
            engine_kwargs.get("amber_trajectory_format", "auto"),
            engine_kwargs.get("amber_executable", "pmemd"),
        )
    return "xtc"


@dataclass
class WalkerTask:
    """A single short MD run: everything a worker needs, sans device binding.

    ``device_index`` is injected by the backend at dispatch time (the local
    backend assigns GPU slots dynamically; schedulers bind GPUs per job).
    """

    index: int
    engine_name: str
    engine_kwargs: dict
    prepare_kwargs: dict
    steps: int
    stride: int
    traj_out: str
    start_coords: Any = None
    # -1 is a sentinel meaning "device binding is managed externally" (a SLURM /
    # PBS array task inherits the scheduler's CUDA_VISIBLE_DEVICES). The local
    # backend overwrites this with a concrete GPU slot (>= 0) at dispatch time.
    device_index: int = -1

    def run_kwargs(self) -> dict:
        return {
            "run_index": self.index,
            "start_coords": self.start_coords,
            "steps": self.steps,
            "traj_out": Path(self.traj_out),
            "stride": self.stride,
            "device_index": self.device_index,
        }


def build_walker_tasks(
    *,
    engine_name: str,
    engine_kwargs: dict,
    prepare_kwargs: dict,
    walkers: list,
    steps: int,
    stride: int,
    outdir: Path,
    iteration: int,
) -> list[WalkerTask]:
    """Construct one :class:`WalkerTask` per walker with deterministic file names."""
    suffix = _traj_suffix(engine_name, engine_kwargs)
    tasks: list[WalkerTask] = []
    for idx, coords in enumerate(walkers):
        traj_out = outdir / f"iteration_{iteration}_{idx}.{suffix}"
        task_engine_kwargs = dict(engine_kwargs)
        base_seed = task_engine_kwargs.get("seed")
        if base_seed is not None:
            walker_seed = (int(base_seed) + int(iteration) * 100003 + int(idx) * 1009) % 2147483647
            if walker_seed == 0:
                walker_seed = 1
            task_engine_kwargs["seed"] = walker_seed
        tasks.append(
            WalkerTask(
                index=idx,
                engine_name=engine_name,
                engine_kwargs=task_engine_kwargs,
                prepare_kwargs=prepare_kwargs,
                steps=steps,
                stride=stride,
                traj_out=str(traj_out),
                start_coords=coords,
            )
        )
    return tasks


def run_walker_task(task: WalkerTask, *, engine_cache: dict | None = None) -> bool:
    """Instantiate the engine in-process and run one production walker.

    If ``engine_cache`` is provided (persistent-worker mode) and the engine
    supports warm reuse, a prepared engine is cached under a key that pins it to
    this exact system *and device*, and re-armed for subsequent walkers instead
    of being rebuilt. The result is identical to a fresh build (the OpenMM engine
    reseeds + reinitializes its Context); the only difference is that the
    expensive Context construction + CUDA JIT is paid once per (system, device)
    rather than once per walker. Engines that do not support warm reuse
    (subprocess GROMACS/Amber) always take the fresh path."""
    import warnings

    warnings.filterwarnings(
        "ignore", message="Non-optimal GB parameters detected for GB model HCT"
    )
    warnings.filterwarnings("ignore", message="Reload offsets from trajectory")

    from trails_md.engines.base import EngineFactory

    walker_seed = task.engine_kwargs.get("seed")
    if walker_seed is not None:
        from trails_md.utils.seeds import SeedManager

        SeedManager(int(walker_seed)).set_seed()

    if engine_cache is not None:
        cached = _warm_engine(task, engine_cache)
        if cached is not None:
            return bool(cached.run_production(**task.run_kwargs()))

    engine = EngineFactory.get(task.engine_name, **task.engine_kwargs)
    engine.prepare(**task.prepare_kwargs)
    return bool(engine.run_production(**task.run_kwargs()))


def _warm_engine_key(task: WalkerTask) -> tuple:
    """Cache key for a reusable engine: same system AND same device, seed excluded.

    The seed is per-walker and is re-armed on reuse, so it must NOT be in the key.
    The device index MUST be, or a walker re-armed on a worker that has since moved
    to a different GPU would silently run on the original device, breaking per-walker
    isolation."""
    def _froze(d: dict) -> tuple:
        return tuple(sorted((k, repr(v)) for k, v in d.items() if k != "seed"))

    return (
        task.engine_name,
        task.device_index,
        _froze(task.engine_kwargs),
        _froze(task.prepare_kwargs),
    )


def _warm_engine(task: WalkerTask, cache: dict):
    """Return a prepared, warm-reusable engine for ``task``, or None to fall back.

    None is returned for engines that do not support warm reuse (subprocess
    GROMACS/Amber), so the caller runs them via the normal fresh path."""
    from trails_md.engines.base import EngineFactory

    key = _warm_engine_key(task)
    engine = cache.get(key)
    if engine is not None:
        engine.rearm_for_walker(task.engine_kwargs.get("seed"))
        return engine

    engine = EngineFactory.get(task.engine_name, **task.engine_kwargs)
    if not getattr(engine, "supports_warm_reuse", False):
        return None  # subprocess engine: no warm benefit, use the fresh path
    engine.prepare(**task.prepare_kwargs)
    engine._warm_reuse = True
    cache[key] = engine
    return engine


class ExecutionBackend(ABC):
    """Strategy interface for executing a batch of walker tasks."""

    @abstractmethod
    def execute(self, tasks: list[WalkerTask]) -> list[bool]:
        """Run all ``tasks`` and return success flags ordered by ``task.index``."""


class ExecutionBackendFactory:
    _backends: dict[str, type[ExecutionBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[ExecutionBackend]) -> None:
        cls._backends[name] = backend_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> ExecutionBackend:
        if name not in cls._backends:
            raise ValueError(
                f"Unknown execution backend: {name!r}. "
                f"Available: {sorted(cls._backends)}"
            )
        return cls._backends[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._backends)
