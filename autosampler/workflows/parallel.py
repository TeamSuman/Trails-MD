"""Per-iteration walker execution.

Thin facade over the pluggable execution backends in
:mod:`autosampler.execution`. Builds one task per walker and dispatches the
batch to the configured backend (local multiprocessing, SLURM, or PBS),
returning per-walker success flags. The local backend preserves the original
GPU-slot scheduling behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from autosampler.execution import build_walker_tasks, make_backend


def run_iteration_parallel(
    engine_name: str,
    engine_kwargs: dict,
    prepare_kwargs: dict,
    walkers: list,
    steps: int,
    stride: int,
    outdir: Path,
    iteration: int,
    max_workers: int = 8,
    gpu_ids: Optional[list[int]] = None,
    execution=None,
) -> list:
    """Execute walker production runs via the configured execution backend."""
    outdir.mkdir(parents=True, exist_ok=True)
    if not walkers:
        return []

    tasks = build_walker_tasks(
        engine_name=engine_name,
        engine_kwargs=engine_kwargs,
        prepare_kwargs=prepare_kwargs,
        walkers=walkers,
        steps=steps,
        stride=stride,
        outdir=outdir,
        iteration=iteration,
    )
    backend = make_backend(execution, gpu_ids=gpu_ids, max_workers=max_workers)
    return backend.execute(tasks)
