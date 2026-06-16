"""Generate output.log exploration summaries for an AutoSampler run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from autosampler.cli import load_config
from autosampler.logs import write_exploration_log


def parse_float_list(value: str) -> list[float]:
    try:
        return [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-bins", type=parse_int_list)
    parser.add_argument("--min-values", type=parse_float_list)
    parser.add_argument("--max-values", type=parse_float_list)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config) if args.config else {}
    n_bins = args.n_bins or config.get("n_bins")
    min_values = args.min_values or config.get("min_values")
    max_values = args.max_values or config.get("max_values")

    if n_bins is None or min_values is None or max_values is None:
        raise SystemExit(
            "Provide --config or all of --n-bins, --min-values, and --max-values."
        )

    output = write_exploration_log(
        run_dir=args.run_dir,
        output=args.output,
        n_bins=n_bins,
        min_values=min_values,
        max_values=max_values,
        append=args.append,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
