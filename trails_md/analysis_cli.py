"""CLI: report kinetics / MSM analysis for a finished or ongoing run.

    trails-md-analyze --run-dir runs/my_run [--outfile fig.png] [--temperature 300]
                      [--config config.yaml] [--tau-ps 2.0] [--discard-fraction 0.5]

Reports whichever is present in the run:
  * Weighted-ensemble kinetics (a source->sink `recycle_target` run): the steady-state
    MFPT via the Hill relation (`MFPT = tau / flux`), with a convergence diagnostic and
    a flux/running-MFPT plot.
  * MSM convergence (an `msm.enabled` run): the multi-panel convergence report.
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
        help="Figure path for the MSM report (default: "
        "<run-dir>/analysis/convergence_report.png).",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run config, used to recover tau = step * dt for the WE MFPT if the run "
        "log is unavailable.",
    )
    parser.add_argument(
        "--tau-ps",
        type=float,
        default=None,
        help="Walker segment length in ps (= step * dt); overrides auto-detection.",
    )
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.5,
        help="Leading fraction of the flux series dropped as pre-steady-state "
        "transient (default 0.5).",
    )
    return parser.parse_args(argv)


def _resolve_tau_ps(args: argparse.Namespace) -> float | None:
    """tau (ps) from --tau-ps, else --config (step*dt), else the run log header."""
    if args.tau_ps is not None:
        return float(args.tau_ps)
    if args.config is not None:
        import yaml

        from trails_md.config import TrailsMDConfig

        cfg = TrailsMDConfig(**yaml.safe_load(args.config.read_text()))
        return float(cfg.spawning.step * cfg.engine.dt)
    from trails_md.analysis.data import load_run_meta

    meta = load_run_meta(args.run_dir)
    if "step" in meta and "dt" in meta:
        return float(meta["step"] * meta["dt"])
    return None


def _report_mfpt(result, run_dir: Path, discard_fraction: float, flux, tau_ps) -> None:
    status = (
        "converged"
        if result.converged
        else "NOT converged (run longer / see the flux plot before trusting this)"
    )
    plateau = "n/a" if result.plateau_ratio is None else f"{result.plateau_ratio:.2f}"
    print("Weighted-ensemble kinetics  (Hill relation:  MFPT = tau / flux)")
    print(f"  MFPT estimate      : {result.mfpt_ns:.4g} ns")
    print(f"  tau (segment)      : {tau_ps:g} ps")
    print(f"  iterations         : {result.n_iterations} "
          f"({result.n_flux_events} with recycled flux)")
    print(f"  discard fraction   : {discard_fraction:g} (leading transient dropped)")
    print(f"  flux plateau ratio : {plateau} (2nd half / 1st half of retained tail)")
    print(f"  status             : {status}")

    from trails_md.analysis.plots import save_flux_convergence

    out = run_dir / "analysis" / "flux_convergence.png"
    try:
        save_flux_convergence(flux, tau_ps, out, discard_fraction)
        print(f"  flux plot          : {out}")
    except ImportError as exc:  # matplotlib optional
        print(f"  (flux plot skipped: {exc})")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.run_dir.is_dir():
        raise SystemExit(f"ERROR: run dir not found: {args.run_dir}")

    from trails_md.analysis.data import load_flux_history, load_msm_series

    did_something = False

    # --- Weighted-ensemble kinetics (rate / MFPT) ---
    flux = load_flux_history(args.run_dir)
    if flux:
        tau_ps = _resolve_tau_ps(args)
        if tau_ps is None:
            raise SystemExit(
                "Found a recycled-flux series but could not determine tau (segment "
                "length). Pass --tau-ps PS (= step * dt) or --config CONFIG."
            )
        from trails_md.spawners.we import steady_state_mfpt

        result = steady_state_mfpt(flux, tau_ps, args.discard_fraction)
        if result.mfpt_ns is None:
            print("Weighted-ensemble run found, but no steady-state flux yet "
                  "(nothing recycled). Run more iterations.")
        else:
            _report_mfpt(result, args.run_dir, args.discard_fraction, flux, tau_ps)
        did_something = True

    # --- MSM convergence report ---
    series = load_msm_series(args.run_dir)
    if series["iterations"].size > 0:
        from trails_md.analysis.plots import plot_convergence_report

        out = plot_convergence_report(
            args.run_dir, outfile=args.outfile, temperature=args.temperature
        )
        print(f"Wrote {out} ({series['iterations'].size} MSM iterations).")
        did_something = True

    if not did_something:
        raise SystemExit(
            f"Nothing to analyze under {args.run_dir}: no recycled-flux series "
            "(kinetics mode: spawn_scheme=we + recycle_target) and no msm.npz "
            "(msm.enabled: true)."
        )


if __name__ == "__main__":
    main()
