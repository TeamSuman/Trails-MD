"""History-window (cumulative vs cycle-local) selection.

PaCS-MD ranks and reseeds from the snapshots of the *current cycle only*
(PaCS-Toolkit: ``self.CVs = results[::skip_frame]`` over one cycle's replicas),
whereas TRAILS-MD spawns from the entire accumulated history. To measure what that
single algorithmic difference is worth, the candidate pool must be restrictable to
the most recent N iterations:

    history_window = None -> full cumulative history (default; current behaviour)
    history_window = 0    -> current iteration only (PaCS-MD-like, cycle-local)
    history_window = N    -> the N most recent historical iterations

The pool is also index-synchronised with the trajectory and frame-record lists built
in core.py, so the window must be applied in the one shared helper rather than at
each call site -- otherwise a spawn index would select a different conformation than
the one whose lineage is recorded.
"""

import numpy as np

from trails_md.spawners.history import pooled_history_iterations


def _history(n_iterations: int, dim: int = 2, frames: int = 4) -> dict:
    return {
        i: {"projection": np.zeros((frames, dim), dtype=float)}
        for i in range(n_iterations)
    }


def test_window_none_keeps_full_cumulative_history():
    """Default behaviour is unchanged: every stored iteration stays eligible."""
    history = _history(5)
    assert pooled_history_iterations(history, target_dim=2) == [0, 1, 2, 3, 4]
    assert pooled_history_iterations(history, target_dim=2, window=None) == [
        0, 1, 2, 3, 4
    ]


def test_window_zero_excludes_all_history():
    """window=0 is the cycle-local (PaCS-MD-like) pool: no historical frames."""
    history = _history(5)
    assert pooled_history_iterations(history, target_dim=2, window=0) == []


def test_window_keeps_only_most_recent_iterations():
    history = _history(5)
    assert pooled_history_iterations(history, target_dim=2, window=1) == [4]
    assert pooled_history_iterations(history, target_dim=2, window=2) == [3, 4]


def test_window_larger_than_history_keeps_everything():
    history = _history(3)
    assert pooled_history_iterations(history, target_dim=2, window=99) == [0, 1, 2]


def test_window_applies_after_the_dimension_filter():
    """Dimension filtering still governs eligibility; the window trims what survives.

    A 1-D initial-trajectory projection at iteration -1 must not consume a slot in a
    2-D window, or the pool would silently shrink.
    """
    history = {
        -1: {"projection": np.zeros((4, 1), dtype=float)},
        0: {"projection": np.zeros((4, 2), dtype=float)},
        1: {"projection": np.zeros((4, 2), dtype=float)},
        2: {"projection": np.zeros((4, 2), dtype=float)},
    }
    assert pooled_history_iterations(history, target_dim=2, window=2) == [1, 2]
    assert pooled_history_iterations(history, target_dim=1, window=2) == [-1]


def test_negative_window_is_rejected():
    """A negative window is a configuration error, not a silent full-history pool."""
    history = _history(3)
    try:
        pooled_history_iterations(history, target_dim=2, window=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a negative window")


# --- wiring: the window must reach the candidate pool and the config ---------


def test_cumulative_points_respects_window():
    """The density spawner's candidate pool shrinks with the window.

    Historical frames are stacked before the current ones, so pool size is the
    observable that decides which frames a spawn index can reach.
    """
    from trails_md.spawners.density import _cumulative_points

    history = _history(3, dim=2, frames=4)  # 3 iterations x 4 frames = 12
    points = np.zeros((4, 2), dtype=float)  # current iteration

    assert _cumulative_points(points, history).shape[0] == 16  # default: all
    assert _cumulative_points(points, history, window=None).shape[0] == 16
    assert _cumulative_points(points, history, window=1).shape[0] == 8
    assert _cumulative_points(points, history, window=0).shape[0] == 4  # cycle-local


def test_density_spawner_honours_history_window():
    """A spawner built with history_window=0 can only return current-iteration frames."""
    from trails_md.spawners.density import DensitySpawner

    history = _history(3, dim=2, frames=4)
    points = np.zeros((4, 2), dtype=float)

    cycle_local = DensitySpawner(
        n_bins=[4, 4], min_values=[-1.0, -1.0], max_values=[1.0, 1.0],
        history_window=0,
    )
    picks = cycle_local.sample(points, top_n=4, history=history)
    assert picks, "spawner returned no frames"
    assert max(picks) < points.shape[0], (
        f"cycle-local spawner reached a historical frame: {picks}"
    )

    cumulative = DensitySpawner(
        n_bins=[4, 4], min_values=[-1.0, -1.0], max_values=[1.0, 1.0],
    )
    assert cumulative.sample(points, top_n=4, history=history) is not None


def test_config_exposes_history_window_defaulting_to_full_history():
    from trails_md.config import SpawningConfig

    assert SpawningConfig().history_window is None
    assert SpawningConfig(history_window=0).history_window == 0


def test_frame_records_stay_index_synced_with_the_windowed_pool():
    """core's frame/lineage mapping must use the same window as the spawner pool.

    pooled_history_iterations is the single source of truth precisely so these two
    cannot diverge. If core kept the full history while the spawner was windowed, a
    spawn index would resolve to a different conformation than the one whose lineage
    is recorded -- silently, with no error anywhere.
    """
    import types

    from tests.test_review_fixes import _bare_core, _iteration_entry
    from trails_md.spawners.density import _historical_points

    sampler = _bare_core()
    sampler.config.spawning.history_window = 0
    sampler.history = {
        0: _iteration_entry(0, n_walkers=2, frames_per_walker=3, n_features=2),
        1: _iteration_entry(1, n_walkers=2, frames_per_walker=3, n_features=2),
    }
    current = np.zeros((6, 2), dtype=float)

    pool = _historical_points(current, sampler.history, window=0)
    records = sampler._sampling_frame_records([], target_dim=2)
    trajectories = sampler._sampling_trajectories([], target_dim=2)

    assert pool.shape[0] == 0, "cycle-local pool should contain no historical frames"
    assert len(records) == pool.shape[0], (
        f"frame records ({len(records)}) desynced from spawner pool ({pool.shape[0]})"
    )
    assert len(trajectories) == 0, "cycle-local run should pool no historical trajectories"


def test_default_config_leaves_frame_records_on_full_history():
    """Regression guard: without a window, core still pools the entire history."""
    from tests.test_review_fixes import _bare_core, _iteration_entry
    from trails_md.spawners.density import _historical_points

    sampler = _bare_core()
    sampler.config.spawning.history_window = None
    sampler.history = {
        0: _iteration_entry(0, n_walkers=2, frames_per_walker=3, n_features=2),
        1: _iteration_entry(1, n_walkers=2, frames_per_walker=3, n_features=2),
    }
    current = np.zeros((6, 2), dtype=float)

    pool = _historical_points(current, sampler.history)
    records = sampler._sampling_frame_records([], target_dim=2)
    assert pool.shape[0] == 12
    assert len(records) == 12
