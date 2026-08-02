"""RiteWeight: randomized iterative trajectory reweighting.

Estimates the stationary (equilibrium or steady-state) distribution from
mis-distributed trajectory data by iteratively reweighting short trajectory
segments until their weighted distribution is self-consistent with the
stationary distribution of a transition matrix built from those same weights.

Reference
---------
S. Kania, R. J. Webber, G. Simpson, D. Aristoff and D. M. Zuckerman,
"Randomized iterative trajectory reweighting for steady-state distributions
without discretization error", PNAS 123, e2529246123 (2026).

This is an independent implementation written from the published algorithm.
The authors' reference implementation (github.com/ZuckermanLab/rite_weight)
carries no licence statement and was deliberately not copied or vendored.

Why this matters for adaptive sampling
--------------------------------------
Adaptive selection deliberately over-samples sparse regions, so the
configurations produced by an exploration campaign are *not* Boltzmann
distributed. RiteWeight corrects exactly that: it fixes mis-*weighting*.
Two consequences follow, and both are limitations worth stating plainly:

* It cannot fix mis-*coverage*. A basin never visited cannot be reweighted
  into existence, and no diagnostic here will tell you one is missing.
* It requires segment pairs drawn from unbiased dynamics. TRAILS-MD applies
  no biasing forces, but velocities are redrawn at each respawn, so pairs
  must never straddle a respawn boundary -- the same constraint that governs
  the time-lagged CV estimators.

The key algorithmic device is that the clustering is *re-randomised every
iteration*. Segments that shared a cluster (and so kept fixed relative
weights) in one iteration are separated in the next, so the fixed point is
governed by the microstate transition matrix rather than by any particular
discretisation. In practice this means the answer should not depend on the
number of clusters -- a property worth asserting in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["riteweight", "RiteWeightResult"]


@dataclass
class RiteWeightResult:
    """Outcome of a RiteWeight run.

    Attributes
    ----------
    weights:
        Final per-segment weights, normalised to sum to one. These are the
        weights averaged over the last ``average_last`` iterations, which
        suppresses the residual noise from the random clustering.
    n_iterations:
        Number of iterations actually performed.
    weight_drift:
        Per-iteration L1 change in the weight vector. A run that has settled
        shows this decreasing and then fluctuating about a small value; a run
        that has not settled does not. Reported so callers can judge
        convergence rather than assume it.
    n_clusters:
        Cluster count used for each random discretisation.
    """

    weights: np.ndarray
    n_iterations: int
    weight_drift: np.ndarray = field(repr=False)
    n_clusters: int = 0

    @property
    def converged(self) -> bool:
        """Heuristic: late drift is well below early drift and small in absolute terms."""
        if len(self.weight_drift) < 20:
            return False
        early = float(np.mean(self.weight_drift[: max(1, len(self.weight_drift) // 10)]))
        late = float(np.mean(self.weight_drift[-max(1, len(self.weight_drift) // 10) :]))
        return late < 0.5 * early and late < 1e-2


def _random_clustering(
    points: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign points to clusters seeded at randomly chosen data points.

    A fresh random seeding each iteration is the mechanism that averages away
    discretisation error, so this deliberately does NOT run k-means to
    convergence -- the clustering is meant to differ between iterations.
    """
    n = len(points)
    k = min(n_clusters, n)
    centre_idx = rng.choice(n, size=k, replace=False)
    centres = points[centre_idx]

    # Exact nearest-centre assignment. A KD-tree over the (few) centres turns the
    # cost from O(n*k) into roughly O(n log k), which is the difference between
    # feasible and not: a production campaign supplies ~10^6 pooled points and
    # thousands of iterations, where the brute-force form needs ~10^11 distance
    # evaluations. The query is exact (k=1 nearest neighbour), so the assignment
    # is identical to brute force -- a test pins that equivalence.
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centres)
        _, labels = tree.query(points, k=1, workers=-1)
        return labels.astype(np.int64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
        labels = np.empty(n, dtype=np.int64)
        chunk = max(1, int(2e7 // max(1, k)))
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            d = np.linalg.norm(points[lo:hi, None, :] - centres[None, :, :], axis=2)
            labels[lo:hi] = np.argmin(d, axis=1)
        return labels


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Left eigenvector of T for eigenvalue 1, normalised to a probability vector.

    Falls back to a uniform distribution if the spectrum is degenerate or the
    dominant eigenvector is unusable (which happens when a random clustering
    yields a disconnected transition graph).
    """
    n = T.shape[0]
    try:
        vals, vecs = np.linalg.eig(T.T)
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    v = np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))])
    v = np.abs(v)  # sign is arbitrary; a probability vector is non-negative
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(n, 1.0 / n)
    return v / s


def riteweight(
    start_features: np.ndarray,
    end_features: np.ndarray,
    *,
    n_clusters: int = 100,
    n_iterations: int = 5000,
    initial_weights: np.ndarray | None = None,
    learning_rate: float = 1.0,
    average_last: int = 1000,
    seed: int | None = None,
) -> RiteWeightResult:
    """Reweight trajectory segments toward the stationary distribution.

    Parameters
    ----------
    start_features, end_features:
        ``(n_segments, n_features)`` arrays holding the featurised first and
        second configuration of each segment. Features must be invariant to
        rotation and translation (pairwise distances, torsions, a learned
        projection, ...). Both configurations of a pair must come from the
        same continuous stretch of unbiased dynamics.
    n_clusters:
        Number of clusters per random discretisation. The fixed point should
        be insensitive to this; it trades resolution against the statistics
        available per cluster.
    n_iterations:
        Reweighting iterations. Each uses a fresh random clustering.
    initial_weights:
        Optional prior over segments; defaults to uniform, the uninformative
        choice. Note the fixed point depends on this prior, so a strongly
        informative and wrong prior can bias the result.
    learning_rate:
        Exponent damping the multiplicative update, ``(pi_I / w_I) ** alpha``.
        1.0 applies the full update; smaller values damp oscillation when the
        per-cluster statistics are sparse.
    average_last:
        Number of trailing iterations to average the weights over, which
        suppresses noise from the random clustering.
    seed:
        Seed for the clustering RNG, for reproducibility.

    Returns
    -------
    RiteWeightResult
    """
    start_features = np.asarray(start_features, dtype=float)
    end_features = np.asarray(end_features, dtype=float)
    if start_features.ndim == 1:
        start_features = start_features.reshape(-1, 1)
    if end_features.ndim == 1:
        end_features = end_features.reshape(-1, 1)
    if start_features.shape != end_features.shape:
        raise ValueError(
            "start_features and end_features must have the same shape; got "
            f"{start_features.shape} and {end_features.shape}"
        )
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1; got {n_clusters}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1; got {n_iterations}")

    n_segments = len(start_features)
    if initial_weights is None:
        weights = np.full(n_segments, 1.0 / n_segments)
    else:
        weights = np.asarray(initial_weights, dtype=float).copy()
        if len(weights) != n_segments:
            raise ValueError("initial_weights must have one entry per segment")
        if np.any(weights <= 0):
            raise ValueError("initial_weights must be strictly positive")
        weights /= weights.sum()

    rng = np.random.default_rng(seed)
    # Cluster over the pooled configurations so that a segment's start and end
    # are discretised on the same footing.
    pooled = np.vstack([start_features, end_features])

    drift = np.empty(n_iterations)
    accum = np.zeros(n_segments)
    n_accum = 0
    keep_from = max(0, n_iterations - average_last)

    for it in range(n_iterations):
        labels = _random_clustering(pooled, n_clusters, rng)
        start_labels = labels[:n_segments]
        end_labels = labels[n_segments:]
        k = int(labels.max()) + 1

        # Weighted transition counts between clusters, then row-normalise.
        counts = np.zeros((k, k))
        np.add.at(counts, (start_labels, end_labels), weights)
        row = counts.sum(axis=1)
        occupied = row > 0
        T = np.zeros((k, k))
        T[occupied] = counts[occupied] / row[occupied, None]
        # Unvisited clusters get a self-transition so T stays a stochastic
        # matrix; they carry no weight and cannot affect the update below.
        T[~occupied, ~occupied] = 1.0

        pi = _stationary_distribution(T)

        # Rescale each cluster's total weight to pi_I, leaving the relative
        # weights of segments inside a cluster untouched (Eq. 2 of the paper).
        w_cluster = np.zeros(k)
        np.add.at(w_cluster, start_labels, weights)
        ratio = np.ones(k)
        good = (w_cluster > 0) & (pi > 0)
        ratio[good] = pi[good] / w_cluster[good]
        if learning_rate != 1.0:
            ratio = ratio**learning_rate

        new_weights = weights * ratio[start_labels]
        total = new_weights.sum()
        if not np.isfinite(total) or total <= 0:
            # Degenerate update (all mass annihilated): keep the previous
            # weights rather than propagate NaNs.
            new_weights = weights.copy()
            total = new_weights.sum()
        new_weights /= total

        drift[it] = float(np.abs(new_weights - weights).sum())
        weights = new_weights

        if it >= keep_from:
            accum += weights
            n_accum += 1

    final = accum / n_accum if n_accum else weights
    final = final / final.sum()
    return RiteWeightResult(
        weights=final,
        n_iterations=n_iterations,
        weight_drift=drift,
        n_clusters=n_clusters,
    )
