"""Regression tests for the July-2026 correctness review fixes.

Covers three confirmed bugs:

* A1 -- ``_collect_msm_trajectories`` sliced the cumulative projection by a
  constant ``step//stride`` and so crossed walker boundaries for engines that
  write a different frame count (GROMACS writes the t=0 frame), injecting
  spurious inter-walker transitions into the MSM.
* A2 -- the spawner pooled historical frames filtered by projection dimension
  while ``core.py`` built the trajectory / frame-record lists without that
  filter, so a spawn index could resolve to the wrong conformation when history
  mixed dimensionalities (e.g. a 2-D initial-trajectory injection alongside an
  n-D adaptive space).
* A3 -- core sampling drew from the global ``random`` module, which an external
  library could desynchronise; draws now come from an instance-bound generator.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from trails_md.spawners.history import pooled_history_iterations, projection_dim


# --------------------------------------------------------------------------- #
# A3 -- instance-bound RNG isolation (light: only needs SeedManager + numpy)
# --------------------------------------------------------------------------- #
def test_seed_manager_rng_is_isolated_from_global_streams():
    """The instance generator is unaffected by global random/numpy consumption."""
    import random

    from trails_md.utils.seeds import SeedManager

    sm1 = SeedManager(7)
    a = sm1.rng.integers(0, 1000, size=16)

    sm2 = SeedManager(7)
    # Perturb both global streams between construction and the draw.
    random.random()
    random.seed(999)
    np.random.seed(123)
    _ = np.random.rand(64)
    b = sm2.rng.integers(0, 1000, size=16)

    np.testing.assert_array_equal(a, b)


def test_seed_manager_rng_state_round_trips():
    """bit_generator.state can be captured and restored (checkpoint mechanism)."""
    from trails_md.utils.seeds import SeedManager

    sm = SeedManager(3)
    sm.rng.integers(0, 10, size=5)  # advance
    saved = sm.rng.bit_generator.state
    expected = sm.rng.integers(0, 10_000, size=8)

    sm.rng.bit_generator.state = saved
    restored = sm.rng.integers(0, 10_000, size=8)
    np.testing.assert_array_equal(expected, restored)


# --------------------------------------------------------------------------- #
# A2 -- shared pooling helper (light)
# --------------------------------------------------------------------------- #
def test_pooled_history_iterations_filters_by_dimension():
    history = {
        -1: {"projection": np.zeros((4, 2))},  # 2-D physical-CV injection
        0: {"projection": np.zeros((6, 5))},  # 5-D adaptive
        1: {"projection": np.zeros((6, 5))},
        2: {"projection": None},  # dropped after retraining
    }
    assert pooled_history_iterations(history, target_dim=5) == [0, 1]
    assert pooled_history_iterations(history, target_dim=2) == [-1]
    assert pooled_history_iterations(history, target_dim=None) == [-1, 0, 1]
    assert projection_dim(np.zeros((3, 5))) == 5
    assert projection_dim(np.zeros(7)) == 1


# --------------------------------------------------------------------------- #
# Spawner resume state -- DensitySpawner.recent_bins must survive a checkpoint
# round-trip so hard-mode bin-avoidance reproduces an uninterrupted run.
# --------------------------------------------------------------------------- #
def test_density_spawner_recent_bins_round_trips():
    from trails_md.spawners.density import DensitySpawner

    spawner = DensitySpawner(probabilistic=False, recent_window=3)
    spawner.recent_bins.append({(0, 1), (2, 3)})  # grid bin ids are tuples
    spawner.recent_bins.append({(4, 5)})

    state = spawner.state_dict()

    restored = DensitySpawner(probabilistic=False, recent_window=3)
    restored.load_state_dict(state)

    assert list(restored.recent_bins) == list(spawner.recent_bins)
    # The set-union used by hard-mode selection must match exactly.
    assert set().union(*restored.recent_bins) == {(0, 1), (2, 3), (4, 5)}


# --------------------------------------------------------------------------- #
# Core-object tests (need MDAnalysis / core import)
# --------------------------------------------------------------------------- #
core = pytest.importorskip("trails_md.core")
from trails_md.paths import build_frame_records  # noqa: E402


def _bare_core(step=1000, stride=100):
    sampler = object.__new__(core.TrailsMDCore)
    sampler.config = types.SimpleNamespace(
        spawning=types.SimpleNamespace(step=step, stride=stride, walker=2),
    )
    sampler.history = {}
    return sampler


def _iteration_entry(iteration, n_walkers, frames_per_walker, n_features, tag=0.0):
    """A history entry mimicking a real run: projection + aligned frame records."""
    total = n_walkers * frames_per_walker
    # Encode the owning walker in every feature column so boundary crossings are
    # detectable: a correct per-walker segment is constant-valued.
    walker_of_row = np.repeat(np.arange(n_walkers), frames_per_walker)
    projection = (walker_of_row[:, None] + tag) * np.ones((total, n_features))
    trajectories = [f"iter{iteration}_w{w}.nc" for w in range(n_walkers)]
    frames = build_frame_records(
        iteration=iteration,
        trajectories=trajectories,
        points=projection,
        walker_parents=[None] * n_walkers,
        expected_frames=frames_per_walker,
    )
    return {
        "projection": projection,
        "frames": frames,
        "trajectories": trajectories,
    }


def test_collect_msm_trajectories_respects_walker_boundaries():
    """A1: GROMACS-style (step//stride + 1) frame counts must not be re-sliced."""
    sampler = _bare_core(step=1000, stride=100)  # constant would give 10
    # Two walkers, each with 11 frames (the t=0 + 10 GROMACS case).
    sampler.history = {
        0: _iteration_entry(0, n_walkers=2, frames_per_walker=11, n_features=1)
    }

    trajs = sampler._collect_msm_trajectories()

    assert len(trajs) == 2, "expected exactly one continuous segment per walker"
    assert all(t.shape[0] == 11 for t in trajs)
    # Each segment belongs to a single walker => constant-valued (no crossing).
    for walker, seg in enumerate(trajs):
        assert np.allclose(seg, walker), "segment crossed a walker boundary"


def test_collect_msm_trajectories_drops_mismatched_dimension():
    """A1: a lower-dimensional injection is excluded from the MSM pool."""
    sampler = _bare_core()
    sampler.history = {
        -1: _iteration_entry(-1, n_walkers=1, frames_per_walker=6, n_features=2),
        0: _iteration_entry(0, n_walkers=2, frames_per_walker=11, n_features=1),
    }
    trajs = sampler._collect_msm_trajectories()
    # Only the 1-D (latest) iteration contributes; the 2-D iter -1 is skipped.
    assert len(trajs) == 2
    assert all(t.shape[1] == 1 for t in trajs)


def test_spawn_index_maps_to_pooled_frame_after_dim_mismatch():
    """A2: frame records and the spawner pool stay index-synchronized."""
    from trails_md.spawners.density import _historical_points

    sampler = _bare_core()
    sampler.history = {
        -1: _iteration_entry(
            -1, n_walkers=1, frames_per_walker=4, n_features=2, tag=7.0
        ),
        0: _iteration_entry(0, n_walkers=2, frames_per_walker=3, n_features=5, tag=0.0),
    }
    current_points = np.ones((6, 5)) * 3.0  # current iteration, 5-D

    # What the spawner actually pools (historical part) for a 5-D projection.
    hist_points = _historical_points(current_points, sampler.history)

    # Fixed path: same dimension filter -> aligned lengths and values.
    records = sampler._sampling_frame_records(
        [], target_dim=projection_dim(current_points)
    )
    assert len(records) == hist_points.shape[0]
    for i in range(hist_points.shape[0]):
        np.testing.assert_allclose(records[i]["cv"], hist_points[i])

    # Buggy path (no filter) would prepend the 4 mismatched iter -1 records,
    # offsetting every spawn index -- assert the filter actually changes length.
    unfiltered = sampler._sampling_frame_records([], target_dim=None)
    assert len(unfiltered) == len(records) + 4
