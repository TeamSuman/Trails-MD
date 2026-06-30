"""Tests for the MSM transition-matrix convergence criterion and the
uncertainty x leverage x flux MSM-guided spawner."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from autosampler.msm import TransitionMatrixCriterion  # noqa: E402
from autosampler.msm.diagnostics import MSMResult  # noqa: E402
from autosampler.spawners import SpawnerFactory  # noqa: E402


def _result(T, pi, counts, *, eigenvectors=None, symbols=None):
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    counts = np.asarray(counts, dtype=float)
    count_matrix = T * counts[:, None]  # row i sums to counts[i]
    return MSMResult(
        lagtime=5,
        n_microstates=n,
        n_states_active=n,
        timescales=np.array([10.0]),
        stationary_distribution=np.asarray(pi, dtype=float),
        transition_matrix=T,
        cluster_centers=np.zeros((n, 1)),
        counts_per_state=counts,
        count_matrix=count_matrix,
        eigenvectors=None if eigenvectors is None else np.asarray(eigenvectors, float),
        state_symbols=np.arange(n) if symbols is None else np.asarray(symbols, int),
    )


# ── TransitionMatrixCriterion ───────────────────────────────────────────────
def test_tmatrix_criterion_satisfied_when_well_sampled():
    T = np.array([[0.9, 0.1], [0.1, 0.9]])
    crit = TransitionMatrixCriterion(tol=0.2)
    status = crit.update(_result(T, [0.5, 0.5], [10000, 10000]))
    assert status.satisfied is True
    assert status.value < 0.2


def test_tmatrix_criterion_unsatisfied_when_sparse():
    T = np.array([[0.9, 0.1], [0.1, 0.9]])
    crit = TransitionMatrixCriterion(tol=0.2)
    status = crit.update(_result(T, [0.5, 0.5], [10, 10]))
    assert status.satisfied is False
    assert status.value > 0.2


def test_tmatrix_criterion_requires_count_matrix():
    crit = TransitionMatrixCriterion()
    res = _result(np.eye(2), [0.5, 0.5], [100, 100])
    res.count_matrix = None
    status = crit.update(res)
    assert status.satisfied is False
    assert "count_matrix" in status.detail


def test_tmatrix_criterion_flux_mask_ignores_tiny_entries():
    # A noisy, barely-visited transition with negligible stationary flux must not
    # block convergence when the high-flux transitions are well determined.
    T = np.array(
        [[0.90, 0.10, 0.00], [0.10, 0.90, 0.00], [0.30, 0.30, 0.40]]
    )
    pi = np.array([0.499, 0.499, 0.002])  # state 2 has negligible weight
    counts = np.array([100000, 100000, 8])  # state 2 poorly sampled
    crit = TransitionMatrixCriterion(tol=0.1, min_flux=1e-2)
    assert crit.update(_result(T, pi, counts)).satisfied is True


# ── MSM-guided spawner ──────────────────────────────────────────────────────
class _FakeClusterModel:
    """Assigns a 1-D point x to microstate floor(x) in [0, 2]."""

    def transform(self, X):
        return np.clip(np.floor(np.asarray(X)[:, 0]).astype(int), 0, 2)


def _three_microstate_result():
    T = np.array(
        [[0.98, 0.01, 0.01], [0.30, 0.40, 0.30], [0.01, 0.01, 0.98]]
    )
    pi = np.array([0.4, 0.2, 0.4])
    counts = np.array([1000.0, 10.0, 1000.0])  # state 1 poorly sampled (high σ)
    eig = np.array([[0.1], [1.0], [0.1]])  # state 1 high leverage
    return _result(T, pi, counts, eigenvectors=eig)


def test_msm_guided_weights_concentrate_on_uncertain_high_leverage_state():
    spawner = SpawnerFactory.get("msm", alpha=1.0, leverage=1, uncertainty=True, seed=0)
    spawner.msm_result = _three_microstate_result()
    spawner.cluster_model = _FakeClusterModel()

    x = np.concatenate([0.5 * np.ones(10), 1.5 * np.ones(10), 2.5 * np.ones(10)])
    cumulative = x.reshape(-1, 1)
    w = spawner._msm_guided_weights(cumulative)
    assert w is not None
    s0, s1, s2 = w[:10].mean(), w[10:20].mean(), w[20:30].mean()
    # The uncertain, high-leverage microstate (1) gets the most weight per frame.
    assert s1 > s0 and s1 > s2


def test_msm_guided_falls_back_to_least_counts_without_msm():
    spawner = SpawnerFactory.get("msm", seed=0)
    assert spawner.msm_result is None
    pts = np.random.default_rng(0).normal(size=(50, 2))
    idx = spawner.sample(pts, top_n=8)
    assert len(idx) == 8
    assert all(0 <= i < 50 for i in idx)


def test_msm_guided_sample_returns_valid_indices():
    spawner = SpawnerFactory.get("msm", alpha=1.0, leverage=1, seed=1)
    spawner.msm_result = _three_microstate_result()
    spawner.cluster_model = _FakeClusterModel()
    x = np.concatenate([0.5 * np.ones(10), 1.5 * np.ones(10), 2.5 * np.ones(10)])
    idx = spawner.sample(x.reshape(-1, 1), top_n=6)
    assert len(idx) == 6 and all(0 <= i < 30 for i in idx)


# ── Estimator populates the new fields ──────────────────────────────────────
def _three_state_chain(n=20000, p=0.02, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([-2.0, 0.0, 2.0])
    P = np.array([[1 - p, p, 0], [p, 1 - 2 * p, p], [0, p, 1 - p]])
    s, states = 0, np.empty(n, int)
    for i in range(n):
        s = rng.choice(3, p=P[s])
        states[i] = s
    return (centers[states] + rng.normal(scale=0.15, size=n)).reshape(-1, 1)


def test_estimator_populates_tmatrix_fields():
    pytest.importorskip("deeptime")
    from autosampler.msm import MSMEstimator

    est = MSMEstimator(lagtime=5, n_microstates=30, n_metastable=3, n_timescales=2)
    res = est.fit([_three_state_chain()])
    assert res.count_matrix is not None
    assert res.state_symbols is not None
    assert res.count_matrix.shape[0] == res.n_states_active
    assert res.eigenvectors is None or res.eigenvectors.shape[0] == res.n_states_active


def test_stable_clustering_runs_and_keeps_cluster_count():
    pytest.importorskip("deeptime")
    from autosampler.msm import MSMEstimator

    est = MSMEstimator(lagtime=5, n_microstates=25, stable_clustering=True)
    traj = _three_state_chain()
    r1 = est.fit([traj])
    r2 = est.fit([np.vstack([traj, _three_state_chain(seed=1)])])
    assert r1.n_microstates == r2.n_microstates  # stable ID space across fits
