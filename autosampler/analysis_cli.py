"""CLI: generate MSM analysis figures from a finished/ongoing run.

    autosampler-analyze --run-dir runs/my_run [--outfile fig.png] [--temperature 300]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output dir.")
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Figure path (default: <run-dir>/analysis/convergence_report.png).",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.run_dir.is_dir():
        raise SystemExit(f"ERROR: run dir not found: {args.run_dir}")

    from autosampler.analysis.data import load_msm_series
    from autosampler.analysis.plots import plot_convergence_report

    series = load_msm_series(args.run_dir)
    if series["iterations"].size == 0:
        raise SystemExit(
            f"No msm.npz found under {args.run_dir}. Was the run started with "
            "msm.enabled: true?"
        )
    out = plot_convergence_report(
        args.run_dir, outfile=args.outfile, temperature=args.temperature
    )
    print(f"Wrote {out} ({series['iterations'].size} MSM iterations).")


if __name__ == "__main__":
    main()
