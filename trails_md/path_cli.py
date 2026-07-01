"""Post-process an Trails-MD run into a connected trajectory path."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trails_md.paths import (
    connected_record_path,
    history_records,
    load_history,
    write_connected_trajectory,
    write_path_metadata,
)


@dataclass(frozen=True)
class PathPair:
    name: str | None
    start: np.ndarray
    end: np.ndarray


def parse_cv(value: str) -> np.ndarray:
    try:
        return np.asarray([float(part) for part in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "CV points must be comma-separated numbers, for example: 1.2,3.4"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--start", type=parse_cv)
    parser.add_argument("--end", type=parse_cv)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help=(
            "Optional JSON or CSV file with path endpoints. JSON must contain a "
            "list of objects with start/end arrays and optional name. CSV may "
            "use start_0,start_1,...,end_0,end_1,... columns or quoted start/end "
            "comma-separated values."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for batch outputs when --pairs-file is used.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON path metadata output. Defaults to OUTPUT.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Checkpoint iteration to read. Defaults to the latest checkpoint.",
    )
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(_normalize_negative_cv_args(raw_argv))
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    batch_mode = args.pairs_file is not None
    if batch_mode:
        if args.output_dir is None:
            parser.error("--output-dir is required when --pairs-file is used")
        if args.metadata is not None:
            parser.error("--metadata is only valid for single-path mode")
        return

    missing = [
        flag
        for flag, value in (
            ("--start", args.start),
            ("--end", args.end),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "single-path mode requires "
            + ", ".join(missing)
            + "; use --pairs-file with --output-dir for batch mode"
        )


def _normalize_negative_cv_args(argv: Sequence[str] | None) -> Sequence[str] | None:
    """Allow --start -1,2 and --end -1,2 despite argparse option parsing."""
    if argv is None:
        return None

    normalized: list[str] = []
    index = 0
    cv_flags = {"--start", "--end"}
    while index < len(argv):
        token = argv[index]
        if token in cv_flags and index + 1 < len(argv):
            normalized.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    history = load_history(args.run_dir, checkpoint=args.checkpoint)
    records = history_records(history)

    if args.pairs_file is not None:
        pairs = _load_path_pairs(args.pairs_file)
        summary: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs):
            stem = _batch_output_stem(index, pair.name)
            output = args.output_dir / f"{stem}.xtc"
            metadata = output.with_suffix(output.suffix + ".json")
            path, start, end = connected_record_path(records, pair.start, pair.end)

            write_connected_trajectory(path, args.topology, output)
            write_path_metadata(path, metadata, start, end)
            summary.append(
                {
                    "name": pair.name,
                    "start": pair.start.tolist(),
                    "end": pair.end.tolist(),
                    "output": str(output),
                    "metadata": str(metadata),
                    "frames": len(path),
                    "start_frame": start.key,
                    "end_frame": end.key,
                }
            )
            print(
                f"Wrote {len(path)} frame(s) for {stem}: "
                f"{start.key} -> {end.key} output={output}"
            )

        summary_path = args.output_dir / "paths_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {len(summary)} path(s). Summary={summary_path}")
        return

    path, start, end = connected_record_path(records, args.start, args.end)

    write_connected_trajectory(path, args.topology, args.output)
    metadata = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    write_path_metadata(path, metadata, start, end)

    print(
        f"Wrote {len(path)} frame(s) from {start.key} to {end.key}: "
        f"{args.output} metadata={metadata}"
    )


def _load_path_pairs(path: Path) -> list[PathPair]:
    if path.suffix.lower() == ".json":
        return _load_json_pairs(path)
    return _load_csv_pairs(path)


def _load_json_pairs(path: Path) -> list[PathPair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Path pairs JSON must contain a list: {path}")

    pairs = [
        PathPair(
            name=str(item["name"]) if item.get("name") is not None else None,
            start=_array_from_sequence(item["start"], f"{path} item {index} start"),
            end=_array_from_sequence(item["end"], f"{path} item {index} end"),
        )
        for index, item in enumerate(payload)
        if isinstance(item, dict)
    ]
    if len(pairs) != len(payload):
        raise ValueError(f"Every path pair JSON item must be an object: {path}")
    return _validate_pairs(pairs, path)


def _load_csv_pairs(path: Path) -> list[PathPair]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        pairs = [
            PathPair(
                name=row.get("name") or None,
                start=_csv_cv(row, "start", path, index),
                end=_csv_cv(row, "end", path, index),
            )
            for index, row in enumerate(reader)
        ]
    return _validate_pairs(pairs, path)


def _csv_cv(row: dict[str, str | None], prefix: str, path: Path, index: int) -> np.ndarray:
    compact = row.get(prefix)
    if compact:
        return parse_cv(compact)

    parts: list[tuple[int, float]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for key, value in row.items():
        match = pattern.match(key)
        if match and value not in (None, ""):
            parts.append((int(match.group(1)), float(value)))
    if not parts:
        raise ValueError(
            f"CSV row {index} in {path} must define {prefix} or {prefix}_0,{prefix}_1,..."
        )
    return np.asarray([value for _, value in sorted(parts)], dtype=float)


def _array_from_sequence(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a list of numbers.")
    return np.asarray([float(part) for part in value], dtype=float)


def _validate_pairs(pairs: list[PathPair], path: Path) -> list[PathPair]:
    if not pairs:
        raise ValueError(f"No path pairs found in {path}")
    for index, pair in enumerate(pairs):
        if pair.start.ndim != 1 or pair.end.ndim != 1:
            raise ValueError(f"Path pair {index} in {path} must use 1D CV arrays.")
        if len(pair.start) != len(pair.end):
            raise ValueError(
                f"Path pair {index} in {path} has start/end dimensionality mismatch: "
                f"{len(pair.start)} vs {len(pair.end)}"
            )
    return pairs


def _batch_output_stem(index: int, name: str | None) -> str:
    suffix = _safe_name(name) if name else f"path_{index:04d}"
    return f"{index:04d}_{suffix}" if name else suffix


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "path"


if __name__ == "__main__":
    main()
