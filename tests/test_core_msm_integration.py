"""Integration test for the MSM wiring inside TrailsMDCore.

Exercises the real ``_collect_msm_trajectories`` and ``_maybe_build_msm``
methods against a synthetic run ``history``, without launching MD. The core
module pulls in heavy MD/ML dependencies at import time, so the whole module is
skipped when they are unavailable.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("deeptime")
pytest.importorskip("torch")
core = pytest.importorskip("trails_md.core")

from trails_md.config import MSMConfig, SpawningConfig  # noqa: E402
from trails_md.msm import MSMEstimator, build_monitor_from_config  # noqa: E402


def _three_state_chain(n_steps, seed, p_escape=0.02):
    rng = np.random.default_rng(seed)
    centers = np.array([-2.0, 0.0, 2.0])
    P = np.array(
        [
            [1 - p_escape, p_escape, 0.0],
            [p_escape, 1 - 2 * p_escape, p_escape],
            [0.0, p_escape, 1 - p_escape],
        ]
    )
    state = 0
    states = np.empty(n_steps, dtype=int)
    for i in range(n_steps):
        state = rng.choice(3, p=P[state])
        states[i] = state
    x = centers[states] + rng.normal(scale=0.15, size=n_steps)
    return x.reshape(-1, 1)


def _make_core(tmp_path, msm_cfg):
    """Build a minimal TrailsMDCore without running its MD-heavy __init__."""
    sampler = object.__new__(core.TrailsMDCore)
    sampler.config = types.SimpleNamespace(
        msm=msm_cfg,
        spawning=SpawningConfig(walker=4, step=2000, stride=10),
    )
    sampler.outdir = tmp_path
    sampler.history = {}
    sampler.iteration = 0
    sampler.converged = False
    sampler.convergence_reason = None
    sampler.last_msm_result = None
    sampler.msm_estimator = MSMEstimator(
        lagtime=msm_cfg.lagtime,
        n_microstates=msm_cfg.n_microstates,
        n_metastable=msm_cfg.n_metastable,
        n_timescales=msm_cfg.n_timescales,
        seed=42,
    )
    sampler.msm_monitor = build_monitor_from_config(msm_cfg)
    return sampler


def _populate_history(sampler, n_iters, frames_per_walker, walkers, seed=0):
    """Fill history with per-iteration projections shaped like a real run."""
    for it in range(n_iters):
        chunks = [
            _three_state_chain(frames_per_walker, seed=seed + it * walkers + w)
            for w in range(walkers)
        ]
        projection = np.vstack(chunks)
        (sampler.outdir / f"iter_{it}").mkdir(parents=True, exist_ok=True)
        sampler.history[it] = {"projection": projection}
    sampler.iteration = n_iters


def test_collect_msm_trajectories_splits_per_walker(tmp_path):
    cfg = MSMConfig(enabled=True, lagtime=5, n_microstates=20, min_frames=10)
    sampler = _make_core(tmp_path, cfg)
    _populate_history(sampler, n_iters=2, frames_per_walker=200, walkers=4)

    trajs = sampler._collect_msm_trajectories()
    # 2 iterations x 4 walkers = 8 continuous trajectories.
    assert len(trajs) == 8
    assert all(t.shape[0] == 200 for t in trajs)


def test_maybe_build_msm_estimates_and_saves(tmp_path):
    cfg = MSMConfig(
        enabled=True,
        lagtime=5,
        n_microstates=40,
        n_metastable=3,
        min_frames=500,
        cadence=1,
    )
    sampler = _make_core(tmp_path, cfg)
    _populate_history(sampler, n_iters=3, frames_per_walker=500, walkers=4)
    sampler.iteration = 3  # current_iteration -> 2

    sampler._maybe_build_msm()

    assert sampler.last_msm_result is not None
    assert sampler.last_msm_result.n_states_active >= 2
    assert (tmp_path / "iter_2" / "msm.npz").exists()
    saved = np.load(tmp_path / "iter_2" / "msm.npz")
    assert "timescales" in saved and "stationary_distribution" in saved


def test_maybe_build_msm_respects_min_frames(tmp_path):
    cfg = MSMConfig(enabled=True, lagtime=5, n_microstates=20, min_frames=10_000)
    sampler = _make_core(tmp_path, cfg)
    _populate_history(sampler, n_iters=1, frames_per_walker=100, walkers=4)
    sampler.iteration = 1

    sampler._maybe_build_msm()
    # Too few frames -> no MSM, no convergence side effects.
    assert sampler.last_msm_result is None
    assert sampler.converged is False


def test_maybe_build_msm_respects_cadence(tmp_path):
    cfg = MSMConfig(enabled=True, lagtime=5, n_microstates=20, min_frames=100, cadence=5)
    sampler = _make_core(tmp_path, cfg)
    _populate_history(sampler, n_iters=2, frames_per_walker=400, walkers=4)
    sampler.iteration = 2  # current_iteration = 1, 1 % 5 != 0 -> skipped

    sampler._maybe_build_msm()
    assert sampler.last_msm_result is None
