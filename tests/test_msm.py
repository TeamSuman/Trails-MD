"""Tests for the MSM subsystem: estimator, diagnostics, convergence monitor."""

from __future__ import annotations

import numpy as np
import pytest

deeptime = pytest.importorskip("deeptime")

from trails_md.msm import (  # noqa: E402
    ConvergenceMonitor,
    ImpliedTimescaleCriterion,
    MSMEstimator,
    MSMResult,
    VAMP2Criterion,
    build_criterion,
)


def _three_state_chain(n_steps=40000, seed=0, p_escape=0.02):
    """Generate a 1D trajectory of a metastable 3-state Markov chain.

    States sit at x = -2, 0, +2 with Gaussian emission so the CV space is
    continuous and must be re-clustered (as in a real run).
    """
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


def test_estimator_recovers_three_states():
    traj = _three_state_chain()
    est = MSMEstimator(lagtime=5, n_microstates=50, n_metastable=3, n_timescales=2)
    result = est.fit([traj])

    assert isinstance(result, MSMResult)
    assert result.n_states_active >= 2
    # Two slow processes resolved and positive.
    assert np.all(result.timescales[np.isfinite(result.timescales)] > 0)
    # Stationary distribution is a probability vector.
    assert result.stationary_distribution.sum() == pytest.approx(1.0, abs=1e-6)
    # PCCA+ recovered three metastable populations that sum to ~1.
    assert result.n_metastable == 3
    assert result.metastable_populations.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.vamp2_score is not None and result.vamp2_score > 1.0


def test_estimator_implied_timescales_sweep():
    traj = _three_state_chain()
    est = MSMEstimator(
        lagtime=5, n_microstates=40, lagtimes=[1, 2, 5, 10], n_timescales=2
    )
    result = est.fit([traj])
    assert result.its is not None
    assert result.its.lagtimes.size >= 2
    assert result.its.timescales.shape[0] == result.its.lagtimes.size


def test_estimator_serialisation_roundtrip():
    traj = _three_state_chain(n_steps=20000)
    est = MSMEstimator(lagtime=5, n_microstates=30, n_metastable=2)
    result = est.fit([traj])
    restored = MSMResult.from_dict(result.to_dict())
    assert restored.lagtime == result.lagtime
    np.testing.assert_allclose(restored.timescales, result.timescales)
    np.testing.assert_allclose(
        restored.stationary_distribution, result.stationary_distribution
    )


def test_estimator_raises_on_empty():
    est = MSMEstimator(lagtime=5, n_microstates=10)
    with pytest.raises(ValueError):
        est.fit([])


def _result(timescales, vamp2=None, populations=None):
    n = len(timescales)
    return MSMResult(
        lagtime=5,
        n_microstates=10,
        n_states_active=n + 1,
        timescales=np.asarray(timescales, dtype=float),
        stationary_distribution=np.ones(n + 1) / (n + 1),
        transition_matrix=np.eye(n + 1),
        cluster_centers=np.zeros((n + 1, 1)),
        counts_per_state=np.ones(n + 1),
        vamp2_score=vamp2,
        n_metastable=None if populations is None else len(populations),
        metastable_populations=None if populations is None else np.asarray(populations),
    )


def test_its_criterion_detects_plateau_and_drift():
    crit = ImpliedTimescaleCriterion(tol=0.05, n_timescales=1)
    assert crit.update(_result([100.0])).satisfied is False  # baseline
    assert crit.update(_result([101.0])).satisfied is True  # 1% change < 5%
    assert crit.update(_result([130.0])).satisfied is False  # 28% change > 5%


def test_vamp2_criterion():
    crit = VAMP2Criterion(tol=0.02)
    assert crit.update(_result([100.0], vamp2=2.0)).satisfied is False
    assert crit.update(_result([100.0], vamp2=2.01)).satisfied is True
    assert crit.update(_result([100.0], vamp2=2.5)).satisfied is False


def test_monitor_converges_after_patience():
    monitor = ConvergenceMonitor(
        [ImpliedTimescaleCriterion(tol=0.05, n_timescales=1)],
        mode="all",
        patience=2,
    )
    assert monitor.update(_result([100.0])) is False  # baseline, not satisfied
    assert monitor.update(_result([100.5])) is False  # satisfied once (streak 1)
    assert monitor.update(_result([100.7])) is True  # satisfied twice -> converged


def test_monitor_does_not_converge_on_drift():
    monitor = ConvergenceMonitor(
        [ImpliedTimescaleCriterion(tol=0.02, n_timescales=1)],
        mode="all",
        patience=2,
    )
    converged = False
    for ts in [100.0, 140.0, 90.0, 160.0, 70.0]:
        converged = monitor.update(_result([ts]))
    assert converged is False


def test_monitor_state_roundtrip():
    monitor = ConvergenceMonitor([VAMP2Criterion(tol=0.1)], patience=3)
    monitor.update(_result([100.0], vamp2=1.0))
    monitor.update(_result([100.0], vamp2=1.01))
    state = monitor.state_dict()
    other = ConvergenceMonitor([VAMP2Criterion(tol=0.1)], patience=3)
    other.load_state_dict(state)
    assert other.streak == monitor.streak


def test_build_criterion_unknown():
    with pytest.raises(ValueError):
        build_criterion("does_not_exist")
