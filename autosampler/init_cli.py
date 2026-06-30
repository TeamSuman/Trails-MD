"""CLI: write a starter AutoSampler input file.

    autosampler-init                 # writes ./config.yaml
    autosampler-init -o my.yaml      # custom path
    autosampler-init --force         # overwrite an existing file

Edit the generated file, then:  autosampler --config config.yaml --check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from autosampler.templates import DEFAULT_TEMPLATE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("config.yaml"), help="Output path."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and not args.force:
        raise SystemExit(
            f"ERROR: {args.output} already exists. Use --force to overwrite."
        )
    args.output.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
    print(f"Wrote starter input file: {args.output}")
    print(f"Next: edit it, then run  autosampler --config {args.output} --check")


if __name__ == "__main__":
    main()
