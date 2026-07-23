"""Regression tests: the MSM lag-time guard-rail and the iteration-0 NaN hint.

Velocity resampling at spawn severs phase-space continuity, so each walker segment is an
independent trajectory and the MSM lag time is capped by the segment length. If the lag
approaches the segment length the implied-timescale plateau cannot be assessed -- and the
convergence monitor, which watches the ITS across *iterations*, would otherwise certify a
systematically underestimated model as "converged".
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from trails_md.core import TrailsMDCore


class _Core(TrailsMDCore):
    """Bare instance: we only exercise the pure guard method."""

    def __init__(self):  # noqa: D107 - deliberately skip the heavy __init__
        pass


@pytest.fixture
def core():
    return _Core()


def _segments(n_segments: int, length: int) -> list[np.ndarray]:
    return [np.zeros((length, 2)) for _ in range(n_segments)]


def test_lag_well_below_segment_length_is_assessable(core):
    # 100-frame segments, lag 10 -> 10x headroom: fine.
    assert core._msm_lag_is_assessable(_segments(8, 100), lagtime=10) is True


def test_lag_exactly_one_fifth_is_assessable(core):
    assert core._msm_lag_is_assessable(_segments(8, 100), lagtime=20) is True


def test_lag_too_close_to_segment_length_is_rejected(core, caplog):
    # lag 50 of a 100-frame segment: only one lagged pair per segment -> cannot see a plateau.
    with caplog.at_level(logging.WARNING):
        assert core._msm_lag_is_assessable(_segments(8, 100), lagtime=50) is False
    assert "plateau cannot be assessed" in caplog.text


def test_guard_uses_the_SHORTEST_segment(core):
    # One truncated walker must not be masked by long ones.
    trajs = _segments(7, 100) + [np.zeros((12, 2))]
    assert core._msm_lag_is_assessable(trajs, lagtime=10) is False


def test_no_trajectories_is_not_assessable(core):
    assert core._msm_lag_is_assessable([], lagtime=5) is False
