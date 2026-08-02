"""RiteWeight trajectory reweighting.

Implements the algorithm of Kania et al., PNAS 123, e2529246123 (2026),
"Randomized iterative trajectory reweighting for steady-state distributions
without discretization error". Written from the published description; the
authors' reference code carries no licence and was not copied.

The decisive test is a discrete Markov chain whose stationary distribution is
exactly calculable: we deliberately draw segment start states from the WRONG
distribution, then require RiteWeight to recover the true stationary weights.
That is ground truth, not a regression baseline.
"""

import numpy as np
import pytest

from trails_md.analysis.riteweight import riteweight


# A 3-state chain with a well-separated stationary distribution.
P_TRUE = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.10, 0.70, 0.20],
        [0.05, 0.25, 0.70],
    ]
)


def exact_stationary(P):
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def make_segments(n=6000, skew=(0.85, 0.10, 0.05), seed=0):
    """Transition pairs from P_TRUE whose START states follow a WRONG distribution.

    Each state is embedded at a well-separated 1-D coordinate so that any sane
    clustering resolves the three states.
    """
    rng = np.random.default_rng(seed)
    starts = rng.choice(3, size=n, p=np.asarray(skew) / np.sum(skew))
    ends = np.array([rng.choice(3, p=P_TRUE[s]) for s in starts])
    coord = np.array([0.0, 10.0, 20.0])
    jitter = 0.05
    x0 = coord[starts] + rng.normal(0, jitter, n)
    x1 = coord[ends] + rng.normal(0, jitter, n)
    return x0.reshape(-1, 1), x1.reshape(-1, 1), starts


def state_weights(weights, starts):
    """Total weight assigned to segments starting in each state."""
    return np.array([weights[starts == s].sum() for s in range(3)])


def test_recovers_exact_stationary_distribution_from_skewed_data():
    """The headline claim: mis-distributed input, correct stationary output."""
    x0, x1, starts = make_segments()
    pi_true = exact_stationary(P_TRUE)

    res = riteweight(x0, x1, n_clusters=8, n_iterations=3000,
                     average_last=500, seed=1)

    got = state_weights(res.weights, starts)
    # Uniform initial weights reproduce the skewed input distribution, so the
    # test only means something if the starting point is genuinely wrong.
    naive = state_weights(np.full(len(starts), 1 / len(starts)), starts)
    assert np.abs(naive - pi_true).max() > 0.25, "test setup is not actually skewed"
    assert np.abs(got - pi_true).max() < 0.05, (
        f"stationary distribution not recovered: got {got}, want {pi_true}"
    )


def test_fixed_point_is_independent_of_cluster_count():
    """Paper's central claim: the fixed point does not depend on n_clusters."""
    x0, x1, starts = make_segments()
    results = {}
    for n_clusters in (4, 12, 40):
        res = riteweight(x0, x1, n_clusters=n_clusters, n_iterations=3000,
                         average_last=500, seed=2)
        results[n_clusters] = state_weights(res.weights, starts)
    ref = results[4]
    for n_clusters, got in results.items():
        assert np.abs(got - ref).max() < 0.05, (
            f"cluster count {n_clusters} changed the fixed point: {got} vs {ref}"
        )


def test_weights_stay_normalised_and_positive():
    x0, x1, _ = make_segments(n=1500)
    res = riteweight(x0, x1, n_clusters=6, n_iterations=300, average_last=50, seed=3)
    assert np.isclose(res.weights.sum(), 1.0), "weights must remain normalised"
    assert (res.weights > 0).all(), "weights must remain strictly positive"
    assert len(res.weights) == len(x0)


def test_already_correct_input_is_left_alone():
    """Idempotence: start from the true distribution and weights should barely move."""
    x0, x1, starts = make_segments(skew=tuple(exact_stationary(P_TRUE)), seed=4)
    n = len(starts)
    res = riteweight(x0, x1, n_clusters=8, n_iterations=1500,
                     average_last=300, seed=5)
    before = state_weights(np.full(n, 1 / n), starts)
    after = state_weights(res.weights, starts)
    assert np.abs(after - before).max() < 0.05, (
        f"already-stationary input was disturbed: {before} -> {after}"
    )


def test_is_reproducible_under_a_fixed_seed():
    x0, x1, _ = make_segments(n=1200)
    a = riteweight(x0, x1, n_clusters=6, n_iterations=400, average_last=100, seed=7)
    b = riteweight(x0, x1, n_clusters=6, n_iterations=400, average_last=100, seed=7)
    np.testing.assert_allclose(a.weights, b.weights)


def test_rejects_mismatched_inputs():
    x0, x1, _ = make_segments(n=100)
    with pytest.raises(ValueError):
        riteweight(x0, x1[:50], n_clusters=4, n_iterations=10)
    with pytest.raises(ValueError):
        riteweight(x0, x1, n_clusters=0, n_iterations=10)


def test_reports_convergence_diagnostics():
    """A user must be able to tell whether the iteration actually settled."""
    x0, x1, _ = make_segments(n=1500)
    res = riteweight(x0, x1, n_clusters=6, n_iterations=800, average_last=200, seed=8)
    assert res.n_iterations == 800
    assert len(res.weight_drift) == 800
    # Drift is the per-iteration change in weights; it must decrease overall.
    early = np.mean(res.weight_drift[:100])
    late = np.mean(res.weight_drift[-100:])
    assert late < early, f"weights not settling: early={early:.3e} late={late:.3e}"


# --- the test that actually exercises the "Randomized" in RiteWeight ---------
#
# The tests above use three well-separated states, which ANY clustering with
# k >= 3 resolves exactly. That leaves no discretisation error to average away,
# so they cannot detect removal of the per-iteration re-clustering -- a mutation
# run confirmed they all still pass with a single fixed clustering.
#
# Here the states are deliberately UNRESOLVABLE by one clustering: 12 microstates
# on a line, discretised into only 4 clusters. With a fixed clustering the
# relative weights of states sharing a cluster are frozen at their (wrong)
# initial values forever. Only re-randomising the clusters lets those ties be
# broken, which is precisely the paper's claim that the fixed point is set by the
# microstate transition matrix and not by the discretisation.

N_STATES = 12


def _line_chain():
    """Nearest-neighbour walk on a line, biased so the stationary law is non-uniform."""
    # Mild bias only. A strong bias concentrates the stationary law on the last
    # state, and if the skewed sampling then barely visits it the test would be
    # demanding a fix for missing COVERAGE -- which RiteWeight explicitly cannot
    # provide. Keep every state well sampled so this isolates mis-WEIGHTING.
    P = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        right = 0.35 if s < N_STATES - 1 else 0.0
        left = 0.30 if s > 0 else 0.0
        P[s, s] = 1.0 - right - left
        if s < N_STATES - 1:
            P[s, s + 1] = right
        if s > 0:
            P[s, s - 1] = left
    return P


def _line_segments(n=20000, seed=11):
    P = _line_chain()
    rng = np.random.default_rng(seed)
    # Wrong distribution, but every state still receives ample statistics:
    # roughly the reverse of the true stationary ordering.
    skew = (0.30 / 0.35) ** np.arange(N_STATES)
    skew /= skew.sum()
    starts = rng.choice(N_STATES, size=n, p=skew)
    ends = np.array([rng.choice(N_STATES, p=P[s]) for s in starts])
    jitter = 0.02
    x0 = starts + rng.normal(0, jitter, n)
    x1 = ends + rng.normal(0, jitter, n)
    return x0.reshape(-1, 1), x1.reshape(-1, 1), starts, exact_stationary(P)


def test_recovers_structure_finer_than_the_clustering():
    """Fewer clusters than states: only re-randomised clustering can succeed."""
    x0, x1, starts, pi_true = _line_segments()
    res = riteweight(x0, x1, n_clusters=4, n_iterations=4000,
                     average_last=800, seed=21)
    got = np.array([res.weights[starts == s].sum() for s in range(N_STATES)])
    naive = np.array([(starts == s).mean() for s in range(N_STATES)])
    # The input must genuinely be wrong, or the test proves nothing.
    assert np.abs(naive - pi_true).max() > 0.10, "test setup is not skewed enough"
    assert np.abs(got - pi_true).max() < 0.05, (
        f"failed to resolve structure finer than the clustering:\n"
        f"  got  {np.round(got, 3)}\n  want {np.round(pi_true, 3)}"
    )


def test_clustering_assigns_each_point_to_its_nearest_centre():
    """Lock the clustering CONTRACT before optimising its implementation.

    The assignment must be exact nearest-centre. This guards the KD-tree
    implementation against silently returning approximate neighbours, which
    would bias the transition matrix in a way no other test would catch.
    """
    from trails_md.analysis.riteweight import _random_clustering

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3))
    labels = _random_clustering(pts, n_clusters=7, rng=np.random.default_rng(1))

    # Recover the centres the routine chose, then verify by brute force.
    centres = np.array([pts[labels == c].mean(axis=0) for c in range(labels.max() + 1)])
    # A point must be at least as close to its own cluster's members as implied;
    # test the invariant directly against the true centre set instead.
    centre_idx = np.random.default_rng(1).choice(len(pts), size=7, replace=False)
    true_centres = pts[centre_idx]
    brute = np.argmin(
        np.linalg.norm(pts[:, None, :] - true_centres[None, :, :], axis=2), axis=1
    )
    np.testing.assert_array_equal(labels, brute)
    assert centres.shape[0] <= 7
