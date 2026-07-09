"""Command line interface for Trails-MD YAML runs."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "trails-md-matplotlib")
)

SYSTEM_PATH_KEYS = (
    "conf_file",
    "top_file",
    "system_file",
    "project_file",
    "trajectory_topology_file",
)


def resolve_config_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve config paths relative to the YAML file location."""
    resolved = copy.deepcopy(config)

    system = resolved.get("system", {})
    for key in SYSTEM_PATH_KEYS:
        value = system.get(key)
        if value and not Path(value).is_absolute():
            system[key] = str((base_dir / value).resolve())

    engine = resolved.get("engine", {})
    amber_input = engine.get("amber_input_file")
    if amber_input and not Path(amber_input).is_absolute():
        engine["amber_input_file"] = str((base_dir / amber_input).resolve())

    outdir = resolved.get("outdir")
    if outdir and not Path(outdir).is_absolute():
        resolved["outdir"] = str((base_dir / outdir).resolve())

    return resolved


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not contain a YAML mapping: {config_path}")
    return resolve_config_paths(config, config_path.parent)


def run(
    config: dict[str, Any],
    iterations: int,
    resume: str | int | None = None,
    ignore_missing_history: bool = False,
) -> tuple[Path, int]:
    from trails_md.core import TrailsMDCore

    sampler = TrailsMDCore(config)
    sampler.validate_preflight()
    sampler.prepare()

    if resume is not None:
        checkpoint_iteration = (
            sampler.latest_checkpoint_iteration() if resume == "latest" else int(resume)
        )
        sampler.restore_checkpoint(
            checkpoint_iteration,
            ignore_missing_history=ignore_missing_history,
        )
        walkers = sampler.resume_walkers()
        print(
            "Resumed from checkpoint "
            f"iter_{checkpoint_iteration}; next iteration is {sampler.iteration}."
        )
    else:
        walkers = sampler.generate_initial_walkers()

    completed_iterations = 0
    for _iteration in range(iterations):
        # run_iteration is the single authority on whether an iteration has
        # enough successful walkers to proceed: it raises when the success rate
        # falls below config.min_success_fraction (1.0 = abort on any failure)
        # and otherwise continues with the survivors.
        result = sampler.run_iteration(walkers)
        completed_iterations += 1
        walkers = result["walkers"]
        if result.get("converged"):
            print(f"Converged: {result.get('convergence_reason')}")
            break

    # Persist a machine-readable campaign outcome next to the outputs so tooling
    # (validators, analysis, an HPC agent benchmarking convergence) does not have
    # to scrape the driver log. Written for every backend since cli.run is the
    # single campaign entry point.
    import json

    try:
        (sampler.outdir / "convergence.json").write_text(
            json.dumps(
                {
                    "converged": bool(getattr(sampler, "converged", False)),
                    "convergence_reason": getattr(sampler, "convergence_reason", None),
                    "completed_iterations": completed_iterations,
                    "requested_iterations": iterations,
                },
                indent=2,
            )
        )
    except OSError as exc:
        logging.warning("Could not write convergence.json: %s", exc)

    return sampler.outdir, completed_iterations


def check_config(config: dict[str, Any]) -> Path:
    from trails_md.core import TrailsMDCore

    sampler = TrailsMDCore(config)
    sampler.validate_preflight()
    return sampler.outdir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML config path. Relative paths inside it are resolved from this file.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume from a checkpoint. Use --resume for the latest checkpoint "
            "or --resume N for checkpoints/iter_N."
        ),
    )
    parser.add_argument(
        "--ignore-missing-history",
        action="store_true",
        help="Ignore missing or unreadable history deltas when restoring a checkpoint.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs and executables, then exit before running MD.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        type=str.upper,
    )
    args = parser.parse_args(argv)
    if args.iterations < 0:
        parser.error("--iterations must be greater than or equal to 0")
    if args.resume not in (None, "latest"):
        try:
            args.resume = int(args.resume)
        except ValueError:
            parser.error("--resume must be omitted, 'latest', or an integer")
        if args.resume < 0:
            parser.error("--resume checkpoint iteration must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        if args.check:
            outdir = check_config(config)
            print(f"Preflight checks passed. Output directory: {outdir}")
            return
        from trails_md.welcome import print_run_banner

        print_run_banner(
            config,
            config_path=config_path,
            iterations=args.iterations,
            color=sys.stdout.isatty(),
        )
        outdir, completed_iterations = run(
            config,
            args.iterations,
            resume=args.resume,
            ignore_missing_history=args.ignore_missing_history,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(f"Completed {completed_iterations} iteration(s). Output: {outdir}")


if __name__ == "__main__":
    main()
