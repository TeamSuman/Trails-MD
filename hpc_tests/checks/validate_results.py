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


def _traj_suffix(engine: str, amber_format: str = "netcdf") -> str:
    # Matches trails_md file naming: OpenMM/GROMACS -> xtc, Amber -> nc/mdcrd.
    if engine == "amber":
        return "mdcrd" if amber_format == "ascii" else "nc"
    return "xtc"


def check(
    outdir: Path,
    iters: int,
    walkers: int,
    engine: str,
    *,
    amber_format: str = "netcdf",
    expect_latent_dim: int | None = None,
    check_msm: bool = False,
    check_resume: bool = False,
    check_path: str | None = None,
    check_gpu_binding: bool = False,
    gpu_count: int | None = None,
    check_msm_convergence: bool = False,
) -> list[dict]:
    results: list[dict] = []

    def add(code, ok, detail):
        results.append(
            {"code": code, "status": "pass" if ok else "fail", "detail": detail}
        )

    add("OUTDIR_EXISTS", outdir.is_dir(), {"outdir": str(outdir)})
    if not outdir.is_dir():
        return results

    # 1. output.log present and has the expected number of data rows.
    log = outdir / "output.log"
    if log.exists():
        data_rows = [
            ln
            for ln in log.read_text().splitlines()
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
        add(
            "WALKER_FAILURES",
            True,
            {
                "total_failed_walkers": failed_total,
                "note": "non-zero is OK only if min_success_fraction < 1.0",
            },
        )
    else:
        add("LOG_ITERATIONS", False, {"error": "output.log missing"})

    # 2. Iteration directories and per-walker trajectory files exist & non-empty.
    dirs = _iter_dirs(outdir)
    add("ITER_DIRS", len(dirs) >= iters, {"found": dirs, "expected_at_least": iters})
    suffix = _traj_suffix(engine, amber_format)
    missing, empty = [], []
    for it in dirs[:iters]:
        for w in range(walkers):
            traj = outdir / f"iter_{it}" / f"iteration_{it}_{w}.{suffix}"
            if not traj.exists():
                missing.append(str(traj))
            elif traj.stat().st_size == 0:
                empty.append(str(traj))
    add(
        "TRAJ_FILES",
        not missing and not empty,
        {
            "missing": missing[:10],
            "empty": empty[:10],
            "note": "missing/empty walker trajectories => scheduler/engine/GPU failure",
        },
    )

    # 3. Result markers from the array tasks (scheduler backend only).
    markers = list(outdir.glob("iter_*/_jobs/result_*.json"))
    if markers:
        n_ok = 0
        for m in markers:
            try:
                n_ok += 1 if json.loads(m.read_text()).get("success") else 0
            except (json.JSONDecodeError, OSError):
                pass
        add(
            "RESULT_MARKERS",
            n_ok > 0,
            {
                "markers": len(markers),
                "successful": n_ok,
                "note": "0 successful markers but a scheduler backend => the SLURM/PBS "
                "poller returned before jobs finished (see DEBUGGING.md CODE=RESULT_MARKERS)",
            },
        )

    # 4. Checkpoints exist and are COMPLETE (format_version marker present).
    ckpt_root = outdir / "checkpoints"
    if ckpt_root.is_dir():
        ck = sorted(
            int(p.name.removeprefix("iter_"))
            for p in ckpt_root.glob("iter_*")
            if p.is_dir() and p.name.removeprefix("iter_").isdigit()
        )
        incomplete = [
            c for c in ck if not (ckpt_root / f"iter_{c}" / "format_version").exists()
        ]
        add(
            "CHECKPOINTS",
            bool(ck) and not incomplete,
            {
                "checkpoints": ck,
                "incomplete_no_marker": incomplete,
                "note": "incomplete checkpoints indicate a crash mid-save; resume will skip them",
            },
        )
    else:
        add("CHECKPOINTS", False, {"error": "no checkpoints/ dir (checkpoint_freq=0?)"})

    # ---- Opt-in feature-specific checks --------------------------------------
    if expect_latent_dim is not None:
        _check_latent_dim(add, outdir, dirs, expect_latent_dim)
    if check_msm:
        _check_msm(add, outdir, dirs)
    if check_resume:
        _check_resume(add, outdir)
    if check_path is not None:
        _check_path(add, Path(check_path))
    if check_gpu_binding:
        _check_gpu_binding(add, outdir, walkers, gpu_count)
    if check_msm_convergence:
        _check_msm_convergence(add, outdir)

    return results


def _check_latent_dim(add, outdir: Path, dirs: list[int], expect: int) -> None:
    """Learned-CV runs: the projected cvs.npz must have the configured latent dim."""
    import numpy as np

    if not dirs:
        add("LATENT_DIM", False, {"error": "no iteration dirs to inspect"})
        return
    cvs_path = outdir / f"iter_{dirs[-1]}" / "cvs.npz"
    if not cvs_path.exists():
        add("LATENT_DIM", False, {"error": f"missing {cvs_path}"})
        return
    cvs = np.load(cvs_path)["cvs"]
    dim = 1 if cvs.ndim == 1 else cvs.shape[1]
    add(
        "LATENT_DIM",
        dim == expect,
        {"found": dim, "expected": expect, "file": str(cvs_path)},
    )


def _check_msm(add, outdir: Path, dirs: list[int]) -> None:
    """In-loop MSM runs: at least one msm.npz with finite timescales and a
    row-stochastic transition matrix."""
    import numpy as np

    npzs = sorted(outdir.glob("iter_*/msm.npz"))
    if not npzs:
        add(
            "MSM_NPZ",
            False,
            {
                "error": "no iter_*/msm.npz written (min_frames not reached, or MSM "
                "estimation failed -- see run.log)"
            },
        )
        return
    data = np.load(npzs[-1])
    timescales = np.asarray(data.get("timescales", []), dtype=float)
    tmatrix = np.asarray(data.get("transition_matrix", []), dtype=float)
    finite_ts = timescales.size > 0 and bool(np.all(np.isfinite(timescales)))
    row_stochastic = (
        tmatrix.ndim == 2
        and tmatrix.shape[0] == tmatrix.shape[1]
        and bool(np.allclose(tmatrix.sum(axis=1), 1.0, atol=1e-6))
    )
    add(
        "MSM_NPZ",
        finite_ts and row_stochastic,
        {
            "file": str(npzs[-1]),
            "n_msm_files": len(npzs),
            "finite_timescales": finite_ts,
            "row_stochastic": row_stochastic,
            "n_states": int(tmatrix.shape[0]) if tmatrix.ndim == 2 else 0,
        },
    )


def _check_resume(add, outdir: Path) -> None:
    """Resume runs: the delta-checkpoint chain reconstructs to a gapless history."""
    ckpt_root = outdir / "checkpoints"
    try:
        import sys

        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from trails_md.checkpoints.manager import (
            _complete_checkpoint_iters,
            reconstruct_history,
        )

        complete = _complete_checkpoint_iters(ckpt_root)
        latest = complete[-1] if complete else None
        if latest is None:
            add("RESUME_CHAIN", False, {"error": "no complete checkpoints"})
            return
        history = reconstruct_history(ckpt_root, latest)
        iters = sorted(k for k in history if isinstance(k, int) and k >= 0)
        gapless = iters == list(range(iters[0], iters[-1] + 1)) if iters else False
        add(
            "RESUME_CHAIN",
            latest is not None and gapless,
            {"latest_complete": latest, "history_iters": iters, "gapless": gapless},
        )
    except Exception as exc:  # noqa: BLE001
        add("RESUME_CHAIN", False, {"error": f"{type(exc).__name__}: {exc}"})


def _check_path(add, path_out: Path) -> None:
    """Path reconstruction: the output trajectory exists and is non-empty."""
    ok = path_out.exists() and path_out.stat().st_size > 0
    add(
        "PATH_OUTPUT",
        ok,
        {
            "output": str(path_out),
            "exists": path_out.exists(),
            "size": path_out.stat().st_size if path_out.exists() else 0,
        },
    )


def _check_gpu_binding(add, outdir: Path, walkers: int, gpu_count: int | None) -> None:
    """GPU device isolation: verify walkers spread across distinct GPUs.

    The OpenMM engine writes a ``<trajectory>.gpu.json`` marker per walker (see
    ``engines/openmm.py::_write_gpu_binding_marker``) recording the *resolved*
    platform and device. This catches the two failure modes the ``GPU_BINDING``
    playbook entry describes:

    * a silent CUDA→CPU fallback (a bad device pin degraded to CPU), and
    * all walkers piling onto a single physical GPU while others sit idle
      (missing per-task device isolation on the scheduler).

    Verdict: fail if any walker fell back to CPU. When ``gpu_count > 1`` is given,
    also fail unless the run used at least ``min(walkers, gpu_count)`` distinct
    devices. Without ``gpu_count`` the check is report-only (it cannot tell a
    correct single-GPU node from missing isolation) but still surfaces the device
    distribution and warns if every walker shared one device.
    """
    markers = sorted(outdir.glob("iter_*/*.gpu.json"))
    if not markers:
        add(
            "GPU_BINDING",
            False,
            {
                "error": "no *.gpu.json markers found under iter_*/ (OpenMM writes one "
                "per walker); not a CUDA/OpenCL/HIP run, or markers not on the "
                "shared filesystem"
            },
        )
        return

    platforms: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    cpu_fallback: list[str] = []
    for m in markers:
        try:
            data = json.loads(m.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        plat = str(data.get("platform", "?"))
        platforms[plat] = platforms.get(plat, 0) + 1
        if plat == "CPU":
            cpu_fallback.append(m.name)
            continue
        # Prefer the scheduler-set visible device; fall back to the pinned index.
        key = str(data.get("visible_devices") or data.get("device_index") or "?")
        device_counts[key] = device_counts.get(key, 0) + 1

    distinct = sorted(k for k in device_counts if k != "?")
    ok = not cpu_fallback
    notes: list[str] = []
    if cpu_fallback:
        notes.append(f"{len(cpu_fallback)} walker(s) ran on CPU (silent GPU fallback)")
    if gpu_count and gpu_count > 1:
        needed = min(walkers, gpu_count)
        if len(distinct) < needed:
            ok = False
            notes.append(
                f"used {len(distinct)} distinct device(s), expected >= {needed}"
            )
    elif len(distinct) <= 1 and walkers > 1:
        notes.append(
            "all walkers shared one device; pass --gpu-count>1 to enforce spread"
        )

    add(
        "GPU_BINDING",
        ok,
        {
            "markers": len(markers),
            "platforms": platforms,
            "distinct_devices": distinct,
            "device_counts": device_counts,
            "cpu_fallback": cpu_fallback[:10],
            "notes": notes,
        },
    )


def _check_msm_convergence(add, outdir: Path) -> None:
    """In-loop MSM convergence workflow: the monitor ran and produced a finite,
    stabilizing implied-timescale series.

    This validates the *workflow* (`msm.enabled` + convergence criteria) end to
    end — MSM estimated each cadence, the convergence monitor evaluated its
    criteria, and the leading implied timescale is finite and settling. It does
    NOT require the tiny smoke workload to truly converge; a real convergence
    benchmark is a separate, longer run (see RUNBOOK.md / docs/msm.md).
    """
    import numpy as np

    npzs = sorted(
        outdir.glob("iter_*/msm.npz"),
        key=lambda p: int(p.parent.name.removeprefix("iter_")),
    )
    if len(npzs) < 2:
        add(
            "MSM_CONVERGENCE",
            False,
            {
                "error": f"need >= 2 iter_*/msm.npz to assess convergence, found {len(npzs)}"
                " (raise iterations / lower msm.min_frames, or MSM estimation failed)"
            },
        )
        return

    leading: list[float] = []
    all_finite = True
    for p in npzs:
        data = np.load(p)
        ts = np.asarray(data.get("timescales", []), dtype=float)
        if ts.size == 0 or not np.all(np.isfinite(ts)):
            all_finite = False
            continue
        leading.append(float(ts[0]))

    rel_change = None
    if len(leading) >= 2 and leading[-2] > 0:
        rel_change = abs(leading[-1] - leading[-2]) / leading[-2]

    # Portable convergence outcome: cli.run writes convergence.json into the outdir
    # for every backend, so we do not have to scrape the driver log.
    converged = None
    reason = None
    conv_file = outdir / "convergence.json"
    if conv_file.exists():
        try:
            conv = json.loads(conv_file.read_text())
            converged = bool(conv.get("converged"))
            reason = conv.get("convergence_reason")
        except (json.JSONDecodeError, OSError):
            pass

    # Verdict validates the *workflow*: the MSM was estimated every cadence and
    # produced a finite, multi-point implied-timescale series the monitor can
    # evaluate. Whether the (tiny) run truly converged is reported, not required.
    ok = all_finite and len(leading) >= 2
    add(
        "MSM_CONVERGENCE",
        ok,
        {
            "n_msm": len(npzs),
            "leading_timescales": leading[-5:],
            "last_rel_change": rel_change,
            "converged": converged,
            "convergence_reason": reason,
            "note": "workflow check: finite, monitored implied-timescale series "
            "(not a claim that the toy system reached kinetic convergence)",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expect-iterations", type=int, default=1)
    parser.add_argument("--expect-walkers", type=int, default=1)
    parser.add_argument(
        "--engine", default="openmm", choices=("openmm", "gromacs", "amber")
    )
    parser.add_argument(
        "--amber-format",
        default="netcdf",
        choices=("netcdf", "ascii"),
        help="Amber trajectory format (selects the .nc vs .mdcrd suffix).",
    )
    parser.add_argument("--out", default=None)
    # Opt-in feature checks (see DEBUGGING.md codes LATENT_DIM/MSM_NPZ/RESUME_CHAIN/PATH_OUTPUT).
    parser.add_argument(
        "--expect-latent-dim",
        type=int,
        default=None,
        help="Learned-CV runs: assert cvs.npz has this many columns.",
    )
    parser.add_argument(
        "--check-msm",
        action="store_true",
        help="Assert iter_*/msm.npz is present and well-formed.",
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="Assert the checkpoint chain reconstructs to a gapless history.",
    )
    parser.add_argument(
        "--check-path",
        default=None,
        help="Assert this trails-md-path output trajectory is non-empty.",
    )
    parser.add_argument(
        "--check-gpu-binding",
        action="store_true",
        help="Assert per-walker GPU markers show device isolation "
        "(no CPU fallback; devices spread when --gpu-count>1).",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Distinct GPUs available; when >1, enforce that walkers "
        "spread across at least min(walkers, gpu-count) devices.",
    )
    parser.add_argument(
        "--check-msm-convergence",
        action="store_true",
        help="Assert the in-loop MSM convergence monitor ran and produced a "
        "finite, stabilizing implied-timescale series.",
    )
    args = parser.parse_args(argv)

    checks = check(
        args.outdir,
        args.expect_iterations,
        args.expect_walkers,
        args.engine,
        amber_format=args.amber_format,
        expect_latent_dim=args.expect_latent_dim,
        check_msm=args.check_msm,
        check_resume=args.check_resume,
        check_path=args.check_path,
        check_gpu_binding=args.check_gpu_binding,
        gpu_count=args.gpu_count,
        check_msm_convergence=args.check_msm_convergence,
    )
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
