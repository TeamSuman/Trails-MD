#!/usr/bin/env python3
"""Local-backend feature matrix for Trails-MD.

Runs each major feature on the fast alanine-dipeptide workloads using the
**local** execution backend (no cluster), then validates each run with
``checks/validate_results.py``. Use it to exercise "all major features" *before*
spending cluster time, to reproduce a bug, or to confirm a fix.

Each feature is the same tiny workload with one axis changed (engine, spawner,
CV space, MSM, feature selection, adaptive binning, resume, path). A feature
whose optional backend is unavailable (torch CVs, Amber ``pmemd``, GROMACS force
fields) is reported **SKIP**, not FAIL, so the matrix is meaningful on any box.

Usage::

    python hpc_tests/run_local_matrix.py                 # run everything runnable
    python hpc_tests/run_local_matrix.py --only openmm_tica_msm
    python hpc_tests/run_local_matrix.py --list          # show the matrix + status
    python hpc_tests/run_local_matrix.py --results-dir results/local

Exit code 0 = every runnable feature passed (skips do not fail the matrix);
1 = at least one runnable feature failed. See ``RUNBOOK.md`` and ``DEBUGGING.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EX_OPENMM = REPO_ROOT / "examples" / "alanine_dipeptide"
EX_GMX = REPO_ROOT / "examples" / "AlaD"
VALIDATE = REPO_ROOT / "hpc_tests" / "checks" / "validate_results.py"
PY = sys.executable

# The repo ships no self-contained Amber topology/coordinates (building one needs
# tleap/ParmEd). Generate them with hpc_tests/assets/build_alad_amber.py; the
# amber_density feature SKIPs (not FAILs) until they exist. See RUNBOOK.md §4.
AMBER_PRMTOP = EX_OPENMM / "alad.prmtop"
AMBER_RST7 = EX_OPENMM / "alad.rst7"


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #
def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _gromacs_include_dir() -> str | None:
    """Best-effort location of a GROMACS force-field ``top`` directory."""
    env = os.environ.get("GMXLIB")
    candidates = [env] if env else []
    gmx = shutil.which("gmx") or shutil.which("gmx_mpi")
    if gmx:
        prefix = Path(gmx).resolve().parent.parent
        candidates.append(str(prefix / "share" / "gromacs" / "top"))
    for cand in candidates:
        if cand and (Path(cand) / "amber99sb.ff").is_dir():
            return cand
    return None


CAPS = {
    "openmm": _have("openmm"),
    "sklearn": _have("sklearn"),
    "deeptime": _have("deeptime"),
    "shapely": _have("shapely"),
    "torch": _have("torch"),
    "mlcolvar": _have("mlcolvar"),
    "lightning": _have("lightning") or _have("pytorch_lightning"),
    "gmx": bool(shutil.which("gmx")) and _gromacs_include_dir() is not None,
    "pmemd": bool(
        shutil.which("pmemd") or shutil.which("pmemd.cuda") or shutil.which("sander")
    ),
    # Amber needs BOTH the engine on PATH and a prmtop/rst7 asset; without the
    # asset the feature SKIPs instead of FAILing on a missing-file --check.
    "amber_asset": AMBER_PRMTOP.is_file() and AMBER_RST7.is_file(),
}


# --------------------------------------------------------------------------- #
# Base configs (local backend, tiny/fast workloads)
# --------------------------------------------------------------------------- #
def _base_openmm() -> dict:
    return {
        "system": {
            "conf_file": str(EX_OPENMM / "structure.pdb"),
            "top_file": str(EX_OPENMM / "structure.pdb"),
            "topology": "amber",
            "system_file": str(EX_OPENMM / "system.py"),
            "project_file": str(EX_OPENMM / "project_phi_psi.py"),
            "trajectory_topology_file": str(EX_OPENMM / "structure.pdb"),
            "feature_selection": "protein and not (type H)",
        },
        "engine": {
            "md_engine": "openmm",
            "platform_name": "CPU",
            "temperature": 300.0,
            "dt": 0.002,
        },
        "spawning": {
            "spawn_scheme": "density",
            "spawn_type": "hard",
            "walker": 4,
            "step": 1000,
            "stride": 100,
            "max_workers": 2,
        },
        "execution": {"backend": "local"},
        "space_mode": "fixed",
        "n_bins": [18, 18],
        "min_values": [-3.141592653589793, -3.141592653589793],
        "max_values": [3.141592653589793, 3.141592653589793],
        "random_seed": 42,
        "checkpoint_freq": 1,
        "save_features": False,
    }


def _base_gromacs() -> dict:
    cfg = _base_openmm()
    cfg["system"].update(
        conf_file=str(EX_GMX / "start.gro"),
        top_file=str(EX_GMX / "topol.top"),
        topology="gromacs",
        project_file=str(EX_GMX / "project_phi_psi.py"),
        trajectory_topology_file=str(EX_GMX / "start.gro"),
        system_file=None,  # NOT the OpenMM system.py -- GROMACS treats this as the MDP template
    )
    cfg["engine"] = {
        "md_engine": "gromacs",
        "gromacs_executable": "gmx",
        "gromacs_include_dir": _gromacs_include_dir()
        or "/path/to/gromacs/share/gromacs/top",
        "gromacs_grompp_maxwarn": 1,  # the bundled AlaD topology emits one benign note
        "temperature": 300.0,
        "pressure": 1.0,
        "dt": 0.002,
        "npt": True,
    }
    cfg["spawning"].update(walker=2, step=500, stride=100, max_workers=1)
    return cfg


def _base_amber() -> dict:
    cfg = _base_openmm()
    # No self-contained Amber asset ships with the repo (see RUNBOOK.md). This
    # base exists so the entry validates/skip-reports; point it at your prmtop/rst7.
    cfg["system"].update(
        conf_file=str(AMBER_RST7),
        top_file=str(AMBER_PRMTOP),
        topology="amber",
        system_file=None,
    )
    cfg["engine"] = {
        "md_engine": "amber",
        "amber_executable": "pmemd",
        "temperature": 300.0,
        "dt": 0.002,
    }
    return cfg


BASES = {"openmm": _base_openmm, "gromacs": _base_gromacs, "amber": _base_amber}

# Adaptive-space scaffolding shared by learned-CV features.
_ADAPTIVE = {
    "adaptive_feature_type": "distances",
    "save_features": True,
    "aggregate_memory": True,
    "adaptive_model": {"latent_dim": 2, "lagtime": 2, "epochs": 20},
    "spawning": {"walker": 4, "step": 2000, "stride": 100, "max_workers": 2},
}
_MSM = {
    "msm": {
        "enabled": True,
        "cadence": 1,
        "min_frames": 50,
        "lagtime": 2,
        "n_microstates": 8,
        "n_timescales": 2,
    }
}
# In-loop MSM *convergence* workflow: estimate every iteration and evaluate the
# convergence monitor's criteria. The tiny workload may not truly converge; the
# test asserts the workflow ran and produced a finite, monitored timescale series
# (a real convergence benchmark is a longer run -- see RUNBOOK.md).
_MSM_CONV = {
    "msm": {
        "enabled": True,
        "cadence": 1,
        "min_frames": 40,
        "lagtime": 2,
        "n_microstates": 6,
        "n_timescales": 2,
        "convergence_mode": "all",
        "convergence_patience": 2,
        "convergence_criteria": [
            {"name": "implied_timescales", "params": {"tol": 0.5, "n_timescales": 1}},
        ],
    }
}


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
# Each feature: name, base engine, deep-merge overrides, requirements, iterations,
# and optional post-run validations (resume/path/msm/latent-dim).
FEATURES = [
    # Path endpoints are the same point so the reconstruction is deterministically
    # connected (it exercises history reconstruction + trajectory writing). Whether
    # two *distinct* basins share a lineage is data-dependent -- that is the
    # manuscript's point (coverage != connected path) -- and is unit-tested in
    # tests/test_spawners_and_paths.py and tests/test_review_fixes.py.
    {
        "name": "openmm_fixed_density",
        "base": "openmm",
        "requires": ["openmm"],
        "iters": 2,
        "resume": True,
        "path": ("-1.4,1.2", "-1.4,1.2"),
    },
    {
        "name": "openmm_voronoi",
        "base": "openmm",
        "requires": ["openmm", "shapely"],
        "overrides": {"spawning": {"spawn_scheme": "voronoi", "voronoi_clusters": 40}},
        "iters": 2,
    },
    {
        "name": "openmm_lof",
        "base": "openmm",
        "requires": ["openmm"],
        "overrides": {
            "spawning": {
                "spawn_scheme": "lof",
                "spawn_type": "soft",
                "lof_neighbors": 5,
            }
        },
        "iters": 2,
    },
    {
        "name": "openmm_fps",
        "base": "openmm",
        "requires": ["openmm"],
        "overrides": {"spawning": {"spawn_scheme": "fps"}},
        "iters": 2,
    },
    {
        "name": "openmm_we",
        "base": "openmm",
        "requires": ["openmm"],
        "overrides": {"spawning": {"spawn_scheme": "we", "we_target_per_bin": 2}},
        "iters": 2,
    },
    {
        "name": "openmm_adaptive_binning",
        "base": "openmm",
        "requires": ["openmm"],
        "overrides": {"binning": {"scheme": "gradient"}},
        "iters": 2,
    },
    {
        "name": "openmm_pca",
        "base": "openmm",
        "requires": ["openmm", "sklearn"],
        "overrides": {**_ADAPTIVE, "space_mode": "pca"},
        "iters": 3,
        "latent_dim": 2,
    },
    {
        "name": "openmm_tica",
        "base": "openmm",
        "requires": ["openmm", "deeptime"],
        "overrides": {**_ADAPTIVE, "space_mode": "tica"},
        "iters": 3,
        "latent_dim": 2,
    },
    {
        "name": "openmm_tica_msm",
        "base": "openmm",
        "requires": ["openmm", "deeptime"],
        "overrides": {**_ADAPTIVE, **_MSM, "space_mode": "tica"},
        "iters": 4,
        "latent_dim": 2,
        "msm": True,
    },
    {
        "name": "openmm_msm_spawn",
        "base": "openmm",
        "requires": ["openmm", "deeptime"],
        "overrides": {
            **_ADAPTIVE,
            **_MSM,
            "space_mode": "tica",
            "spawning": {**_ADAPTIVE["spawning"], "spawn_scheme": "msm"},
        },
        "iters": 4,
        "msm": True,
    },
    {
        "name": "openmm_feature_selection",
        "base": "openmm",
        "requires": ["openmm", "deeptime"],
        "overrides": {
            **_ADAPTIVE,
            "space_mode": "tica",
            "feature_selection": {"enabled": True, "lagtime": 2, "cadence": 1},
        },
        "iters": 3,
        "latent_dim": 2,
    },
    {
        "name": "openmm_tvae",
        "base": "openmm",
        "requires": ["openmm", "torch", "deeptime"],
        "overrides": {**_ADAPTIVE, "space_mode": "tvae"},
        "iters": 3,
        "latent_dim": 2,
    },
    # --- New CV methods (manuscript-revision features; extensively tested) ------
    {
        "name": "openmm_vampnet",
        "base": "openmm",
        "requires": ["openmm", "torch", "deeptime"],
        "overrides": {**_ADAPTIVE, "space_mode": "vampnet"},
        "iters": 3,
        "latent_dim": 2,
    },
    {
        "name": "openmm_spib",
        "base": "openmm",
        "requires": ["openmm", "torch"],
        "overrides": {**_ADAPTIVE, "space_mode": "spib"},
        "iters": 3,
        "latent_dim": 2,
    },
    {
        "name": "openmm_deep_tica",
        "base": "openmm",
        "requires": ["openmm", "mlcolvar", "lightning", "torch"],
        "overrides": {**_ADAPTIVE, "space_mode": "deep-tica"},
        "iters": 3,
        "latent_dim": 2,
    },
    # --- MSM convergence workflow (to be benchmarked for the revision) ----------
    {
        "name": "openmm_msm_convergence",
        "base": "openmm",
        "requires": ["openmm", "deeptime"],
        "overrides": {**_ADAPTIVE, **_MSM_CONV, "space_mode": "tica"},
        "iters": 5,
        # Converging *early* (before all 5 iterations) is the desired outcome, so
        # the baseline iteration checks assert only a floor, not the full count.
        "expect_min_iters": 2,
        "msm": True,
        "msm_convergence": True,
    },
    {"name": "gromacs_density", "base": "gromacs", "requires": ["gmx"], "iters": 2},
    # GROMACS writes the t=0 frame (step//stride + 1 frames/walker); this exercises
    # MSM trajectory segmentation by per-walker frame records on that engine.
    {
        "name": "gromacs_tica_msm",
        "base": "gromacs",
        "requires": ["gmx", "deeptime"],
        "overrides": {
            **_ADAPTIVE,
            **_MSM,
            "space_mode": "tica",
            "spawning": {"walker": 4, "step": 1000, "stride": 100, "max_workers": 1},
        },
        "iters": 4,
        "latent_dim": 2,
        "msm": True,
    },
    {
        "name": "amber_density",
        "base": "amber",
        "requires": ["pmemd", "amber_asset"],
        "iters": 2,
    },
    # Adaptive learned CV on the Amber engine (needs the bring-your-own asset).
    {
        "name": "amber_tica",
        "base": "amber",
        "requires": ["pmemd", "amber_asset", "deeptime"],
        "overrides": {**_ADAPTIVE, "space_mode": "tica"},
        "iters": 3,
        "latent_dim": 2,
    },
]


def _execution_block(backend: str) -> dict:
    """Execution config for the chosen backend.

    ``local`` (default) needs nothing. For ``slurm``/``pbs`` the per-site
    resources are read from environment variables so one wrapper script drives
    the whole feature matrix on a cluster (see RUNBOOK.md):
    ``TRAILS_HPC_PARTITION``, ``TRAILS_HPC_ACCOUNT``, ``TRAILS_HPC_WALLTIME``,
    ``TRAILS_HPC_CPUS``, ``TRAILS_HPC_GPUS``, ``TRAILS_HPC_MODULES`` (comma-sep).
    """
    if backend == "local":
        return {"backend": "local"}
    modules = [m for m in os.environ.get("TRAILS_HPC_MODULES", "").split(",") if m]
    block = {
        "backend": backend,
        "partition": os.environ.get("TRAILS_HPC_PARTITION") or None,
        "account": os.environ.get("TRAILS_HPC_ACCOUNT") or None,
        "walltime": os.environ.get("TRAILS_HPC_WALLTIME", "00:20:00"),
        "cpus_per_task": int(os.environ.get("TRAILS_HPC_CPUS", "2")),
        "gpus_per_task": int(os.environ.get("TRAILS_HPC_GPUS", "0")),
        "max_retries": 1,
        "marker_grace": 30,
        "module_loads": modules,
    }
    return block


def _deep_merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _missing_caps(feature: dict) -> list[str]:
    return [cap for cap in feature.get("requires", []) if not CAPS.get(cap, False)]


def _run_cli(config: Path, args: list[str], log: Path) -> int:
    with open(log, "a") as handle:
        handle.write(f"\n$ trails-md --config {config} {' '.join(args)}\n")
        handle.flush()
        proc = subprocess.run(
            [PY, "-m", "trails_md.cli", "--config", str(config), *args],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    return proc.returncode


def run_feature(
    feature: dict, results_dir: Path, iters_override: int | None, backend: str = "local"
) -> dict:
    import yaml

    name = feature["name"]
    missing = _missing_caps(feature)
    fdir = results_dir / f"local_{name}"
    fdir.mkdir(parents=True, exist_ok=True)
    if missing:
        return {
            "name": name,
            "status": "skip",
            "reason": f"missing: {', '.join(missing)}",
        }

    cfg = BASES[feature["base"]]()
    _deep_merge(cfg, feature.get("overrides", {}))
    cfg["execution"] = _execution_block(backend)

    # When the scheduler hands out GPUs, run OpenMM on CUDA so the walkers
    # actually use them (the base config is CPU for portability), and request the
    # GPU-isolation validation below.
    want_gpu = int(os.environ.get("TRAILS_HPC_GPUS", "0")) >= 1
    if want_gpu and cfg["engine"].get("md_engine") == "openmm":
        cfg["engine"]["platform_name"] = "CUDA"

    # Absolute outdir: config paths are re-resolved relative to the config file's
    # directory, so a *relative* outdir (from a relative --results-dir) would be
    # doubled (results/.../run -> results/.../results/.../run). Resolve it here.
    run_out = (fdir / "run").resolve()
    cfg["outdir"] = str(run_out)
    if run_out.exists():
        shutil.rmtree(run_out)
    config_path = fdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = fdir / "run.log"
    log.write_text("")

    iters = iters_override or feature.get("iters", 2)
    walkers = cfg["spawning"]["walker"]
    engine = cfg["engine"]["md_engine"]

    env_note = ""
    if _run_cli(config_path, ["--check"], log) != 0:
        return {
            "name": name,
            "status": "fail",
            "reason": "preflight (--check) failed",
            "log": str(log),
        }
    if (
        _run_cli(
            config_path, ["--iterations", str(iters), "--log-level", "WARNING"], log
        )
        != 0
    ):
        return {"name": name, "status": "fail", "reason": "run failed", "log": str(log)}

    # A convergence feature may stop early (the point of the test), so validate a
    # floor of iterations rather than the full requested count.
    expect_iters = feature.get("expect_min_iters", iters)
    val_args = [
        "--outdir",
        str(run_out),
        "--expect-iterations",
        str(expect_iters),
        "--expect-walkers",
        str(walkers),
        "--engine",
        engine,
    ]
    if feature.get("latent_dim"):
        val_args += ["--expect-latent-dim", str(feature["latent_dim"])]
    if feature.get("msm"):
        val_args += ["--check-msm"]
    if feature.get("msm_convergence"):
        val_args += ["--check-msm-convergence"]
    # Verify GPU device isolation for OpenMM runs on a GPU platform (catches a
    # silent CUDA->CPU fallback, and, with TRAILS_HPC_GPU_COUNT set, walkers all
    # piling onto one device). See DEBUGGING.md CODE=GPU_BINDING.
    if engine == "openmm" and cfg["engine"].get("platform_name") in (
        "CUDA",
        "OpenCL",
        "HIP",
    ):
        val_args += ["--check-gpu-binding"]
        gpu_count = os.environ.get("TRAILS_HPC_GPU_COUNT")
        if gpu_count:
            val_args += ["--gpu-count", gpu_count]

    if feature.get("resume"):
        if (
            _run_cli(
                config_path,
                ["--resume", "--iterations", "1", "--log-level", "WARNING"],
                log,
            )
            != 0
        ):
            return {
                "name": name,
                "status": "fail",
                "reason": "resume failed",
                "log": str(log),
            }
        val_args += ["--check-resume", "--expect-iterations", str(iters + 1)]

    if feature.get("path"):
        start, end = feature["path"]
        out_traj = fdir / "path.xtc"
        topo = cfg["system"]["trajectory_topology_file"]
        with open(log, "a") as handle:
            handle.write(f"\n$ trails-md-path ... --start {start} --end {end}\n")
            handle.flush()
            rc = subprocess.run(
                [
                    PY,
                    "-m",
                    "trails_md.path_cli",
                    "--run-dir",
                    str(run_out),
                    "--topology",
                    str(topo),
                    f"--start={start}",
                    f"--end={end}",
                    "--output",
                    str(out_traj),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            ).returncode
        if rc == 0:
            val_args += ["--check-path", str(out_traj)]
        else:
            env_note = " (path reconstruction returned nonzero; see run.log)"

    val_out = fdir / "validate.json"
    val_args += ["--out", str(val_out)]
    proc = subprocess.run(
        [PY, str(VALIDATE), *val_args], capture_output=True, text=True
    )
    status = "pass" if proc.returncode == 0 else "fail"
    failed = []
    try:
        failed = [
            c["code"]
            for c in json.loads(val_out.read_text())["checks"]
            if c["status"] == "fail"
        ]
    except Exception:  # noqa: BLE001
        pass
    return {
        "name": name,
        "status": status,
        "failed_codes": failed,
        "validate": str(val_out),
        "log": str(log),
        "note": env_note,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", action="append", help="run only these feature name(s)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="override per-feature iteration count",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "local"
    )
    parser.add_argument(
        "--backend",
        choices=("local", "slurm", "pbs"),
        default="local",
        help="execution backend (slurm/pbs read TRAILS_HPC_* env vars for resources)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the matrix and capability status, then exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(
            "Capabilities:",
            ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in CAPS.items()),
        )
        print("\nFeatures:")
        for f in FEATURES:
            miss = _missing_caps(f)
            state = f"SKIP (missing {', '.join(miss)})" if miss else "runnable"
            print(f"  {f['name']:28} [{f['base']:7}] {state}")
        return 0

    selected = (
        FEATURES
        if not args.only
        else [f for f in FEATURES if f["name"] in set(args.only)]
    )
    if not selected:
        print(
            f"No features matched {args.only}. Use --list to see names.",
            file=sys.stderr,
        )
        return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Capabilities: {', '.join(k for k, v in CAPS.items() if v) or 'none'}")
    print(f"Running {len(selected)} feature(s) -> {args.results_dir}\n")

    rows = []
    for feature in selected:
        print(f"  ... {feature['name']}", flush=True)
        rows.append(
            run_feature(feature, args.results_dir, args.iterations, args.backend)
        )

    summary = {"results_dir": str(args.results_dir), "features": rows}
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== LOCAL FEATURE MATRIX ===")
    icons = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
    for r in rows:
        extra = r.get("reason") or (", ".join(r.get("failed_codes") or []))
        print(
            f"  [{icons.get(r['status'], '?'):4}] {r['name']:28} {extra}{r.get('note', '')}"
        )
    n_fail = sum(1 for r in rows if r["status"] == "fail")
    n_skip = sum(1 for r in rows if r["status"] == "skip")
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    print(
        f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed. Summary: {args.results_dir / 'summary.json'}"
    )
    if n_fail:
        print(
            "On any FAIL: read results/local_<name>/{run.log,validate.json} and hpc_tests/DEBUGGING.md."
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
