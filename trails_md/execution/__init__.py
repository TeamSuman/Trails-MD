"""Pluggable execution backends: local multiprocessing and HPC schedulers."""

# Import backend modules for their factory-registration side effects.
from . import (
    local,  # noqa: F401,E402
    pbs,  # noqa: F401,E402
    slurm,  # noqa: F401,E402
)
from .base import (
    ExecutionBackend,
    ExecutionBackendFactory,
    WalkerTask,
    build_walker_tasks,
    run_walker_task,
)

__all__ = [
    "ExecutionBackend",
    "ExecutionBackendFactory",
    "WalkerTask",
    "build_walker_tasks",
    "run_walker_task",
    "make_backend",
]


def make_backend(execution_config, *, gpu_ids=None, max_workers: int = 8):
    """Construct the configured execution backend.

    ``execution_config`` is an ``ExecutionConfig`` (or ``None`` for local
    defaults). ``gpu_ids`` / ``max_workers`` come from the engine/spawning
    config and apply to the local backend.
    """
    if execution_config is None:
        return ExecutionBackendFactory.get(
            "local", gpu_ids=gpu_ids, max_workers=max_workers
        )

    backend = getattr(execution_config, "backend", "local")
    if backend == "local":
        return ExecutionBackendFactory.get(
            "local",
            gpu_ids=gpu_ids,
            max_workers=max_workers,
            walker_timeout=getattr(execution_config, "walker_timeout", None),
            persistent_workers=getattr(execution_config, "persistent_workers", False),
        )

    cfg = execution_config.model_dump()
    cfg.pop("backend", None)
    return ExecutionBackendFactory.get(backend, **cfg)
