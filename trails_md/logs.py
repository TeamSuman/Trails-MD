"""Run-log generation utilities for completed Trails-MD runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_exploration_log(
    *,
    run_dir: Path,
    output: Path | None = None,
    n_bins: list[int],
    min_values: list[float],
    max_values: list[float],
    append: bool = False,
) -> Path:
    output = output or run_dir / "output.log"
    output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append and output.exists() else "w"
    seen: set[tuple[int, ...]] = set()
    cumulative_frames = 0
    total_bins = int(np.prod(n_bins))

    with output.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            _write_header(handle, run_dir, n_bins, min_values, max_values)

        for iter_dir in _iteration_dirs(run_dir):
            iteration = int(iter_dir.name.split("_", 1)[1])
            cvs_path = iter_dir / "cvs.npz"
            if not cvs_path.exists():
                continue

            cvs = np.asarray(np.load(cvs_path)["cvs"], dtype=float)
            bins = _cv_bins(cvs, n_bins, min_values, max_values)
            seen.update(tuple(row) for row in bins)
            cumulative_frames += len(cvs)
            checkpoint_dir = run_dir / "checkpoints" / f"iter_{iteration}"
            exploration_fraction = len(seen) / total_bins if total_bins else 0.0
            row = [
                iteration,
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                len(cvs),
                cumulative_frames,
                len(seen),
                total_bins,
                f"{exploration_fraction:.8f}",
                "NA",
                str(cvs_path),
                str(iter_dir),
                str(checkpoint_dir) if checkpoint_dir.exists() else "NA",
            ]
            handle.write("\t".join(str(value) for value in row) + "\n")

    return output


def _write_header(
    handle: Any,
    run_dir: Path,
    n_bins: list[int],
    min_values: list[float],
    max_values: list[float],
) -> None:
    lines = [
        "# Trails-MD run log",
        f"# outdir={run_dir}",
        "# source=postprocess",
        f"# n_bins={json.dumps(n_bins)}",
        f"# min_values={json.dumps(min_values)}",
        f"# max_values={json.dumps(max_values)}",
        (
            "iteration\trunner_s\tanalysis_s\ttotal_s\t"
            "successful_walkers\tfailed_walkers\tframes_this_iteration\t"
            "cumulative_frames\toccupied_bins\ttotal_bins\texploration_fraction\t"
            "spawn_indices\tcvs_file\ttrajectory_dir\tcheckpoint_dir"
        ),
    ]
    handle.write("\n".join(lines) + "\n")


def _iteration_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in run_dir.glob("iter_*")
            if path.is_dir() and path.name.split("_", 1)[1].isdigit()
        ],
        key=lambda path: int(path.name.split("_", 1)[1]),
    )


def _cv_bins(
    cvs: np.ndarray,
    n_bins: list[int],
    min_values: list[float],
    max_values: list[float],
) -> np.ndarray:
    n_bins_array = np.asarray(n_bins, dtype=int)
    min_array = np.asarray(min_values, dtype=float)
    max_array = np.asarray(max_values, dtype=float)
    if cvs.ndim != 2:
        raise ValueError(f"Expected 2D CV array, got shape {cvs.shape}.")
    if cvs.shape[1] != len(n_bins_array):
        raise ValueError(
            f"CV dimensionality mismatch: cvs has {cvs.shape[1]} columns, "
            f"n_bins has {len(n_bins_array)} values."
        )

    scaled = (cvs - min_array) / (max_array - min_array)
    bins = np.floor(scaled * n_bins_array).astype(int)
    return np.clip(bins, 0, n_bins_array - 1)
