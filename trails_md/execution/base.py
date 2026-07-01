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
    device_index: int = 0

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
        tasks.append(
            WalkerTask(
                index=idx,
                engine_name=engine_name,
                engine_kwargs=engine_kwargs,
                prepare_kwargs=prepare_kwargs,
                steps=steps,
                stride=stride,
                traj_out=str(traj_out),
                start_coords=coords,
            )
        )
    return tasks


def run_walker_task(task: WalkerTask) -> bool:
    """Instantiate the engine in-process and run one production walker."""
    import warnings

    warnings.filterwarnings(
        "ignore", message="Non-optimal GB parameters detected for GB model HCT"
    )
    warnings.filterwarnings("ignore", message="Reload offsets from trajectory")

    from trails_md.engines.base import EngineFactory

    engine = EngineFactory.get(task.engine_name, **task.engine_kwargs)
    engine.prepare(**task.prepare_kwargs)
    return bool(engine.run_production(**task.run_kwargs()))


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
