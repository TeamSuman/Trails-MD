#!/usr/bin/env python3
"""Trails-MD HPC preflight checker.

Runs *before* submitting any walker job and records a structured JSON report of
the runtime environment so a human (or a future automated agent) can tell an
environment problem apart from a code problem when a cluster run fails.

Usage::

    python hpc_tests/checks/preflight.py --scheduler slurm --gpu \
        --config hpc_tests/configs/alad_cpu_slurm.yaml \
        --out results/preflight_slurm_cpu.json

Exit code 0 = all *required* checks passed; 1 = a required check failed.
Optional checks (e.g. a GPU on a login node) never fail the run; they are
reported as ``"status": "warn"``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Allow running in-tree (before `pip install`): add the repo root to sys.path so
# `import trails_md` resolves whether or not the package is installed.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def check_python_and_package() -> dict:
    info = {"name": "trails_md_import", "required": True}
    try:
        import trails_md  # noqa: F401
        from trails_md.execution import ExecutionBackendFactory

        info["status"] = "pass"
        info["detail"] = {
            "python": sys.version.split()[0],
            "backends": ExecutionBackendFactory.available(),
        }
    except Exception as exc:  # noqa: BLE001
        info["status"] = "fail"
        info["detail"] = f"cannot import trails_md: {exc!r}"
    return info


def check_scheduler(scheduler: str) -> dict:
    submit = {"slurm": "sbatch", "pbs": "qsub"}[scheduler]
    poll = {"slurm": "squeue", "pbs": "qstat"}[scheduler]
    cancel = {"slurm": "scancel", "pbs": "qdel"}[scheduler]
    found = {tool: bool(shutil.which(tool)) for tool in (submit, poll, cancel)}
    return {
        "name": f"{scheduler}_tools_on_path",
        "required": True,
        "status": "pass" if all(found.values()) else "fail",
        "detail": found,
    }


def check_engines() -> dict:
    """Report which MD engines are actually runnable in this environment."""
    engines = {}
    # OpenMM: importable + platforms
    try:
        import openmm  # noqa: F401
        from openmm import Platform

        platforms = [
            Platform.getPlatform(i).getName()
            for i in range(Platform.getNumPlatforms())
        ]
        engines["openmm"] = {"available": True, "platforms": platforms}
    except Exception as exc:  # noqa: BLE001
        engines["openmm"] = {"available": False, "reason": str(exc)}
    # GROMACS / Amber: look for common executables on PATH
    for tool in ("gmx", "gmx_mpi", "pmemd", "pmemd.cuda", "sander"):
        engines[tool] = {"on_path": bool(shutil.which(tool))}
    return {
        "name": "md_engines",
        "required": False,  # only the engine your config uses must be present
        "status": "pass",
        "detail": engines,
    }


def check_gpu(expect_gpu: bool) -> dict:
    rc, out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    detail = {
        "nvidia_smi_rc": rc,
        "nvidia_smi": out.splitlines()[:8],
        "CUDA_VISIBLE_DEVICES": visible,
    }
    if not expect_gpu:
        return {"name": "gpu", "required": False, "status": "skip", "detail": detail}
    status = "pass" if rc == 0 else "warn"  # login nodes often lack GPUs
    return {"name": "gpu", "required": False, "status": status, "detail": detail}


def check_config(config_path: str | None) -> dict:
    if not config_path:
        return {"name": "config", "required": False, "status": "skip", "detail": None}
    info = {"name": "config_validates", "required": True}
    try:
        import yaml

        from trails_md.cli import resolve_config_paths
        from trails_md.config import TrailsMDConfig
        from pathlib import Path

        with open(config_path) as fh:
            raw = yaml.safe_load(fh)
        resolved = resolve_config_paths(raw, Path(config_path).parent)
        cfg = TrailsMDConfig(**resolved)
        info["status"] = "pass"
        info["detail"] = {
            "backend": cfg.execution.backend,
            "walker": cfg.spawning.walker,
            "gpus_per_task": cfg.execution.gpus_per_task,
            "max_in_flight": cfg.execution.max_in_flight,
            "min_success_fraction": cfg.min_success_fraction,
        }
    except Exception as exc:  # noqa: BLE001
        info["status"] = "fail"
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


def check_shared_filesystem() -> dict:
    """The scheduler backend needs a shared FS visible to compute + submit host."""
    cwd = os.getcwd()
    marker = os.path.join(cwd, f".trails_fs_check_{os.getpid()}")
    ok = False
    detail = {"cwd": cwd}
    try:
        with open(marker, "w") as fh:
            fh.write("ok")
        os.replace(marker, marker + ".final")
        ok = os.path.exists(marker + ".final")
    except OSError as exc:
        detail["error"] = str(exc)
    finally:
        for p in (marker, marker + ".final"):
            try:
                os.unlink(p)
            except OSError:
                pass
    detail["writable"] = ok
    detail["note"] = "Confirm this path is on a shared FS (Lustre/GPFS/NFS), not node-local /tmp."
    return {"name": "filesystem", "required": True, "status": "pass" if ok else "fail", "detail": detail}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("slurm", "pbs"), required=True)
    parser.add_argument("--gpu", action="store_true", help="this run targets a GPU queue")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="write JSON report here")
    args = parser.parse_args(argv)

    checks = [
        check_python_and_package(),
        check_scheduler(args.scheduler),
        check_shared_filesystem(),
        check_engines(),
        check_gpu(args.gpu),
        check_config(args.config),
    ]
    required_failed = [c for c in checks if c.get("required") and c["status"] == "fail"]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler": args.scheduler,
        "gpu_run": args.gpu,
        "hostname": os.uname().nodename,
        "overall": "fail" if required_failed else "pass",
        "checks": checks,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 1 if required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
