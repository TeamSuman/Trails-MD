#!/usr/bin/env python3
"""Validate the output of a Trails-MD HPC test run.

Parses a run ``outdir`` and checks the invariants an HPC scheduler run must
satisfy, then emits a structured JSON verdict plus a human-readable summary.
Designed to be run *after* a cluster job finishes so a person — or an automated
agent debugging the run — can quickly localize a failure.

Usage::

    python hpc_tests/checks/validate_results.py \
        --outdir runs/hpc_alad_cpu_slurm \
        --expect-iterations 5 \
        --expect-walkers 8 \
        --engine openmm \
        --out results/validate_slurm_cpu.json

Exit code 0 = all checks passed, 1 = at least one failed. See DEBUGGING.md for
how to act on each failure code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _iter_dirs(outdir: Path) -> list[int]:
    nums = []
    for p in outdir.glob("iter_*"):
        suffix = p.name.removeprefix("iter_")
        if p.is_dir() and suffix.isdigit():
            nums.append(int(suffix))
    return sorted(nums)


def _traj_suffix(engine: str) -> str:
    # Matches trails_md file naming: OpenMM/GROMACS -> xtc, Amber -> nc/mdcrd.
    if engine == "amber":
        return "nc"  # netcdf default for pmemd.cuda; adjust if using ascii
    return "xtc"


def check(outdir: Path, iters: int, walkers: int, engine: str) -> list[dict]:
    results: list[dict] = []

    def add(code, ok, detail):
        results.append({"code": code, "status": "pass" if ok else "fail", "detail": detail})

    add("OUTDIR_EXISTS", outdir.is_dir(), {"outdir": str(outdir)})
    if not outdir.is_dir():
        return results

    # 1. output.log present and has the expected number of data rows.
    log = outdir / "output.log"
    if log.exists():
        data_rows = [
            ln for ln in log.read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("iteration\t")
        ]
        add(
            "LOG_ITERATIONS",
            len(data_rows) >= iters,
            {"rows": len(data_rows), "expected_at_least": iters, "log": str(log)},
        )
        # Parse failed-walker counts from the log (column 6 = failed_walkers).
        failed_total = 0
        for ln in data_rows:
            cols = ln.split("\t")
            if len(cols) > 5 and cols[5].isdigit():
                failed_total += int(cols[5])
        add("WALKER_FAILURES", True, {"total_failed_walkers": failed_total,
            "note": "non-zero is OK only if min_success_fraction < 1.0"})
    else:
        add("LOG_ITERATIONS", False, {"error": "output.log missing"})

    # 2. Iteration directories and per-walker trajectory files exist & non-empty.
    dirs = _iter_dirs(outdir)
    add("ITER_DIRS", len(dirs) >= iters, {"found": dirs, "expected_at_least": iters})
    suffix = _traj_suffix(engine)
    missing, empty = [], []
    for it in dirs[:iters]:
        for w in range(walkers):
            traj = outdir / f"iter_{it}" / f"iteration_{it}_{w}.{suffix}"
            if not traj.exists():
                missing.append(str(traj))
            elif traj.stat().st_size == 0:
                empty.append(str(traj))
    add("TRAJ_FILES", not missing and not empty,
        {"missing": missing[:10], "empty": empty[:10],
         "note": "missing/empty walker trajectories => scheduler/engine/GPU failure"})

    # 3. Result markers from the array tasks (scheduler backend only).
    markers = list(outdir.glob("iter_*/_jobs/result_*.json"))
    if markers:
        n_ok = 0
        for m in markers:
            try:
                n_ok += 1 if json.loads(m.read_text()).get("success") else 0
            except (json.JSONDecodeError, OSError):
                pass
        add("RESULT_MARKERS", n_ok > 0,
            {"markers": len(markers), "successful": n_ok,
             "note": "0 successful markers but a scheduler backend => the SLURM/PBS "
                     "poller returned before jobs finished (see DEBUGGING.md CODE=RESULT_MARKERS)"})

    # 4. Checkpoints exist and are COMPLETE (format_version marker present).
    ckpt_root = outdir / "checkpoints"
    if ckpt_root.is_dir():
        ck = sorted(int(p.name.removeprefix("iter_")) for p in ckpt_root.glob("iter_*")
                    if p.is_dir() and p.name.removeprefix("iter_").isdigit())
        incomplete = [c for c in ck if not (ckpt_root / f"iter_{c}" / "format_version").exists()]
        add("CHECKPOINTS", bool(ck) and not incomplete,
            {"checkpoints": ck, "incomplete_no_marker": incomplete,
             "note": "incomplete checkpoints indicate a crash mid-save; resume will skip them"})
    else:
        add("CHECKPOINTS", False, {"error": "no checkpoints/ dir (checkpoint_freq=0?)"})

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expect-iterations", type=int, default=1)
    parser.add_argument("--expect-walkers", type=int, default=1)
    parser.add_argument("--engine", default="openmm", choices=("openmm", "gromacs", "amber"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    checks = check(args.outdir, args.expect_iterations, args.expect_walkers, args.engine)
    failed = [c for c in checks if c["status"] == "fail"]
    report = {
        "outdir": str(args.outdir),
        "overall": "fail" if failed else "pass",
        "n_failed": len(failed),
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    for c in checks:
        print(f"  [{c['status'].upper():4}] {c['code']}")
    if failed:
        print("\nFAILED codes ->", ", ".join(c["code"] for c in failed))
        print("See hpc_tests/DEBUGGING.md for the playbook keyed by these codes.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
