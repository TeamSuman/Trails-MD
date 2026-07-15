"""Persistent-worker warm reuse must be bit-for-bit identical to a fresh build.

Rebuilding the OpenMM Context (+ CUDA JIT) every walker is the dominant per-walker
cost for short segments, so a persistent worker caches a prepared engine and
re-arms it. Correctness hinges on one non-obvious fact (verified here on the CPU
platform, so no GPU is needed): re-arming reproduces a fresh build *exactly* only
because the engine reseeds the integrator and then reinitializes the Context. A
naive reseed without reinitialize silently keeps the previous thermostat RNG
stream and yields a different, still-deterministic-looking trajectory -- the kind
of bug that would corrupt a rate estimate without ever crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")
import MDAnalysis as mda  # noqa: E402

from trails_md.execution.base import WalkerTask, run_walker_task  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alanine_dipeptide"


def _task(index: int, seed: int, traj_out: Path) -> WalkerTask:
    from openmm.app import PDBFile

    positions = PDBFile(str(EXAMPLE / "structure.pdb")).getPositions(asNumpy=True)
    return WalkerTask(
        index=index,
        engine_name="openmm",
        engine_kwargs={"platform_name": "CPU", "dt": 0.002, "seed": seed},
        prepare_kwargs={
            "conf": EXAMPLE / "structure.pdb",
            "top": EXAMPLE / "structure.pdb",
            "system_file": EXAMPLE / "system.py",
        },
        steps=200,
        stride=50,
        traj_out=str(traj_out),
        start_coords=positions,
        device_index=0,
    )


def _run(task: WalkerTask, cache: dict | None) -> np.ndarray:
    assert run_walker_task(task, engine_cache=cache) is True
    u = mda.Universe(str(EXAMPLE / "structure.pdb"), task.traj_out)
    return np.stack([ts.positions.copy() for ts in u.trajectory])


def test_warm_reuse_matches_fresh_build_bitwise(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")  # single thread -> deterministic FP

    fresh = _run(_task(0, seed=11, traj_out=tmp_path / "fresh.xtc"), cache=None)

    # A persistent worker: first call builds+caches, second RE-ARMS the warm engine.
    cache: dict = {}
    _run(_task(0, seed=11, traj_out=tmp_path / "warmup.xtc"), cache)  # primes the cache
    assert len(cache) == 1
    warm = _run(_task(0, seed=11, traj_out=tmp_path / "warm.xtc"), cache)

    assert warm.shape == fresh.shape
    np.testing.assert_array_equal(warm, fresh)


def test_warm_reuse_reseeds_per_walker(tmp_path, monkeypatch):
    """A re-armed engine must honour the NEW walker's seed, not the cached one."""
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")

    fresh22 = _run(_task(1, seed=22, traj_out=tmp_path / "fresh22.xtc"), cache=None)

    cache: dict = {}
    _run(_task(0, seed=11, traj_out=tmp_path / "warm11.xtc"), cache)  # seed 11 first
    warm22 = _run(_task(1, seed=22, traj_out=tmp_path / "warm22.xtc"), cache)  # reseed 22

    # Same cached Context, different seed -> must equal a fresh seed-22 run.
    np.testing.assert_array_equal(warm22, fresh22)


def test_different_seeds_diverge(tmp_path, monkeypatch):
    """Sanity: the comparison above is not passing because seeds are ignored."""
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    a = _run(_task(0, seed=11, traj_out=tmp_path / "a.xtc"), cache=None)
    b = _run(_task(0, seed=99, traj_out=tmp_path / "b.xtc"), cache=None)
    assert not np.allclose(a, b)


def test_cache_pins_device_index(tmp_path, monkeypatch):
    """Re-arming must not reuse a Context built for a different device."""
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    from trails_md.execution.base import _warm_engine_key

    t0 = _task(0, seed=11, traj_out=tmp_path / "d0.xtc")
    t1 = _task(0, seed=11, traj_out=tmp_path / "d1.xtc")
    t1.device_index = 1
    assert _warm_engine_key(t0) != _warm_engine_key(t1)
