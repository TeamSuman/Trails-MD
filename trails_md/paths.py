"""Utilities for reconstructing connected paths through sampled trajectories."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameRef:
    iteration: int
    walker: int
    frame: int
    trajectory: str
    cv: tuple[float, ...]
    parent: str | None = None

    @property
    def key(self) -> str:
        return frame_key(self.iteration, self.walker, self.frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "iteration": self.iteration,
            "walker": self.walker,
            "frame": self.frame,
            "trajectory": self.trajectory,
            "cv": list(self.cv),
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameRef:
        return cls(
            iteration=int(data["iteration"]),
            walker=int(data["walker"]),
            frame=int(data["frame"]),
            trajectory=str(data["trajectory"]),
            cv=tuple(float(value) for value in data["cv"]),
            parent=data.get("parent"),
        )


def frame_key(iteration: int, walker: int, frame: int) -> str:
    return f"{iteration}:{walker}:{frame}"


def trajectory_frame_counts(
    trajectories: Iterable[str], expected_frames: int | None = None
) -> list[int]:
    counts: list[int] = []
    for trajectory in trajectories:
        traj_str = str(trajectory)
        if traj_str.endswith(".xtc"):
            try:
                from MDAnalysis.coordinates.XTC import XTCReader  # type: ignore

                with XTCReader(traj_str) as reader:
                    counts.append(int(reader.n_frames))
            except Exception:
                if expected_frames is None:
                    raise
                counts.append(int(expected_frames))
        else:
            if expected_frames is None:
                raise ValueError(
                    f"Cannot determine frame count for non-XTC trajectory {traj_str} without expected_frames."
                )
            counts.append(int(expected_frames))
    return counts


def build_frame_records(
    *,
    iteration: int,
    trajectories: list[str],
    points: np.ndarray,
    walker_parents: list[str | None],
    expected_frames: int | None = None,
) -> list[dict[str, Any]]:
    counts = trajectory_frame_counts(trajectories, expected_frames=expected_frames)
    if sum(counts) != len(points):
        raise ValueError(
            "Trajectory frame count does not match CV count: "
            f"{sum(counts)} frames vs {len(points)} CV rows."
        )

    records: list[dict[str, Any]] = []
    point_offset = 0
    for walker, (trajectory, count) in enumerate(zip(trajectories, counts, strict=False)):
        for frame in range(count):
            parent = (
                walker_parents[walker]
                if frame == 0 and walker < len(walker_parents)
                else frame_key(iteration, walker, frame - 1)
                if frame > 0
                else None
            )
            record = FrameRef(
                iteration=iteration,
                walker=walker,
                frame=frame,
                trajectory=str(trajectory),
                cv=tuple(float(value) for value in np.asarray(points[point_offset])),
                parent=parent,
            )
            records.append(record.to_dict())
            point_offset += 1
    return records


def map_global_frame(records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index < 0 or index >= len(records):
        raise IndexError(f"Frame index {index} is outside the sampled history.")
    return records[index]


def load_history(
    run_dir: Path, checkpoint: int | None = None, ignore_missing: bool = False
) -> dict[int, Any]:
    from trails_md.checkpoints.manager import reconstruct_history

    checkpoint_root = run_dir / "checkpoints"
    if checkpoint is None:
        checkpoint_iters = [
            int(path.name.removeprefix("iter_"))
            for path in checkpoint_root.glob("iter_*")
            if path.is_dir() and path.name.removeprefix("iter_").isdigit()
        ]
        if not checkpoint_iters:
            raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")
        target = max(checkpoint_iters)
    else:
        target = checkpoint
        if not (checkpoint_root / f"iter_{target}").exists():
            raise FileNotFoundError(
                f"History file not found: {checkpoint_root / f'iter_{target}'}"
            )

    # History is delta-checkpointed: each iter_*/history.pkl holds only the
    # entries since the previous checkpoint. Merge them back into the full
    # history (otherwise the lineage/path tools see only the last window).
    return reconstruct_history(checkpoint_root, target, ignore_missing=ignore_missing)


def history_records(history: dict[int, Any]) -> list[FrameRef]:
    records: list[FrameRef] = []
    for iteration in sorted(history):
        entry = history[iteration]
        if not isinstance(entry, dict):
            continue
        for record in entry.get("frames", []):
            records.append(FrameRef.from_dict(record))
    if not records:
        raise ValueError(
            "No frame lineage records found. Re-run sampling with this version of "
            "Trails-MD to generate connected path metadata."
        )
    return records


def nearest_record(records: list[FrameRef], point: np.ndarray) -> FrameRef:
    if not records:
        raise ValueError("Cannot search an empty frame history.")
    cvs = np.asarray([record.cv for record in records], dtype=float)
    if cvs.shape[1] != len(point):
        raise ValueError(
            f"CV dimensionality mismatch: history has {cvs.shape[1]} dimensions, "
            f"query has {len(point)}."
        )
    distances = np.linalg.norm(cvs - np.asarray(point, dtype=float), axis=1)
    return records[int(np.argmin(distances))]


def connected_record_path(
    records: list[FrameRef], start_point: np.ndarray, end_point: np.ndarray
) -> tuple[list[FrameRef], FrameRef, FrameRef]:
    by_key = {record.key: record for record in records}
    start = nearest_record(records, start_point)
    end = nearest_record(records, end_point)

    start_to_root = _lineage_to_root(start, by_key)
    end_to_root = _lineage_to_root(end, by_key)
    start_positions = {record.key: idx for idx, record in enumerate(start_to_root)}

    common_key = None
    end_common_index = None
    for idx, record in enumerate(end_to_root):
        if record.key in start_positions:
            common_key = record.key
            end_common_index = idx
            break
    if common_key is None or end_common_index is None:
        raise ValueError("Start and end frames do not share a recorded lineage.")

    start_common_index = start_positions[common_key]
    path = start_to_root[: start_common_index + 1]
    path.extend(reversed(end_to_root[:end_common_index]))
    return path, start, end


def write_connected_trajectory(
    records: list[FrameRef], topology: str | Path, output: str | Path
) -> None:
    try:
        import MDAnalysis as mda  # type: ignore
    except ImportError:
        _write_connected_trajectory_with_gromacs(records, topology, output)
        return

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    universes: dict[str, Any] = {}
    try:
        first = _universe_for_record(records[0], topology, universes)
        with mda.Writer(str(output), n_atoms=first.atoms.n_atoms) as writer:
            for record in records:
                universe = _universe_for_record(record, topology, universes)
                universe.trajectory[record.frame]
                writer.write(universe.atoms)
    finally:
        for universe in universes.values():
            universe.trajectory.close()


def write_path_metadata(
    records: list[FrameRef],
    output: str | Path,
    start: FrameRef,
    end: FrameRef,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start_frame": start.to_dict(),
        "end_frame": end.to_dict(),
        "frames": [record.to_dict() for record in records],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _lineage_to_root(record: FrameRef, records: dict[str, FrameRef]) -> list[FrameRef]:
    lineage = [record]
    seen = {record.key}
    current = record
    while current.parent is not None:
        parent = records.get(current.parent)
        if parent is None:
            break
        if parent.key in seen:
            raise ValueError(f"Cycle detected in frame lineage at {parent.key}.")
        lineage.append(parent)
        seen.add(parent.key)
        current = parent
    return lineage


def _universe_for_record(
    record: FrameRef, topology: str | Path, universes: dict[str, Any]
) -> Any:
    trajectory = str(record.trajectory)
    universe = universes.get(trajectory)
    if universe is None:
        import MDAnalysis as mda  # type: ignore

        if trajectory.endswith((".crd", ".mdcrd", ".trj")):
            universe = mda.Universe(str(topology), trajectory, format="TRJ")
        else:
            universe = mda.Universe(str(topology), trajectory)
        universes[trajectory] = universe
    return universe


def _write_connected_trajectory_with_gromacs(
    records: list[FrameRef], topology: str | Path, output: str | Path
) -> None:
    if not records:
        raise ValueError("Cannot write a connected trajectory with no frames.")

    gmx = shutil.which("gmx")
    if gmx is None:
        # Allow an explicit override for sites where gmx is not on PATH.
        override = os.environ.get("TRAILS_MD_GMX")
        if override and Path(override).exists():
            gmx = override
    if gmx is None:
        raise ImportError(
            "MDAnalysis is not installed and no GROMACS 'gmx' executable was found. "
            "Install MDAnalysis, put 'gmx' on PATH, or set TRAILS_MD_GMX to its path."
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    time_cache: dict[str, tuple[float, float]] = {}

    with tempfile.TemporaryDirectory(prefix="trails-md-path-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        frame_paths: list[Path] = []
        for index, record in enumerate(records):
            first_time, timestep = time_cache.setdefault(
                record.trajectory, _gromacs_xtc_times(gmx, record.trajectory)
            )
            dump_time = first_time + record.frame * timestep
            frame_path = tmpdir_path / f"frame_{index:06d}.xtc"
            _run_gromacs(
                [
                    gmx,
                    "trjconv",
                    "-s",
                    str(topology),
                    "-f",
                    record.trajectory,
                    "-dump",
                    f"{dump_time:.9f}",
                    "-o",
                    str(frame_path),
                ],
                input_text="0\n",
            )
            frame_paths.append(frame_path)

        output_timestep = time_cache[records[0].trajectory][1]
        concatenated = tmpdir_path / "concatenated.xtc"
        _run_gromacs(
            [
                gmx,
                "trjcat",
                "-f",
                *[str(path) for path in frame_paths],
                "-o",
                str(concatenated),
                "-cat",
            ]
        )
        _run_gromacs(
            [
                gmx,
                "trjconv",
                "-s",
                str(topology),
                "-f",
                str(concatenated),
                "-o",
                str(output),
                "-t0",
                "0",
                "-timestep",
                f"{output_timestep:.9f}",
            ],
            input_text="0\n",
        )


def _gromacs_xtc_times(gmx: str, trajectory: str) -> tuple[float, float]:
    result = _run_gromacs([gmx, "check", "-f", trajectory])
    text = result.stdout + result.stderr
    first_match = re.search(r"Reading frame\s+0\s+time\s+([0-9.+\-Ee]+)", text)
    timestep_match = re.search(r"Time\s+\d+\s+([0-9.+\-Ee]+)", text)
    if first_match is None or timestep_match is None:
        raise RuntimeError(f"Could not parse GROMACS frame times for {trajectory}")
    return float(first_match.group(1)), float(timestep_match.group(1))


def _run_gromacs(
    command: list[str], input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        check=True,
    )
