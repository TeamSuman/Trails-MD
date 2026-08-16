# Reweighting an adaptive ensemble

Adaptive sampling deliberately over-samples sparse regions. The configurations an
exploration campaign produces are therefore **not** Boltzmann distributed, and a
histogram of them is not a free-energy surface. Something has to put the weights back.

Trails-MD offers two routes, and they fail in different ways — which is the reason for
keeping both.

| | MSM reweighting | RiteWeight |
| --- | --- | --- |
| Assumption | Markovian at the chosen lag, on the chosen discretisation | none of either |
| Needs a lag time | **yes** — and the answer depends on it | no |
| Discretisation error | enters through the clustering | averaged away by re-randomising it |
| Cost | one clustering + one eigenproblem | one clustering + one eigenproblem **per iteration** |

Neither fixes mis-*coverage*. A basin that was never visited cannot be reweighted into
existence, and no diagnostic in either method will tell you one is missing.

## MSM reweighting

The standard route: discretise, count transitions at lag $\tau$, solve for the
stationary distribution $\pi$, and give each frame weight $\pi_I / N_I$ where $I$ is its
state and $N_I$ that state's frame count. See [MSM & kinetic seeding](msm.md).

The catch is the lag. Too short and the cluster-level dynamics are not Markovian, so
$\pi$ is wrong; too long and the transition counts run out. Trails-MD reports an
implied-timescale diagnostic precisely so this can be checked rather than assumed.

## RiteWeight

`trails_md.analysis.riteweight` implements the randomized iterative trajectory
reweighting scheme of Kania, Webber, Simpson, Aristoff and Zuckerman,
*PNAS* **123**, e2529246123 (2026). Short trajectory segments are reweighted
iteratively until their weighted distribution is self-consistent with the stationary
distribution of a transition matrix built from those same weights.

The key device is that **the clustering is re-randomised at every iteration**. Segments
that shared a cluster in one iteration are separated in the next, so the fixed point is
governed by the underlying microstate dynamics rather than by any particular
discretisation. In practice the answer should not depend on the cluster count — a
property worth checking, and one the test suite asserts.

```python
import numpy as np
from trails_md.analysis import riteweight

# One (start, end) pair per short segment. Features must be invariant to rotation
# and translation: pairwise distances, torsion sin/cos, a learned projection, ...
result = riteweight(start_features, end_features, n_clusters=150,
                    n_iterations=2000, average_last=400, seed=0)

print(result.converged)          # heuristic: late drift << early drift, and small
weights = result.weights         # one weight per segment, summing to 1

H, xe, ye = np.histogram2d(cv[:, 0], cv[:, 1], bins=60, weights=weights)
F = -kT * np.log(H / H.sum())
```

### Forming the segment pairs

Both configurations of a pair must come from the **same continuous stretch of unbiased
dynamics**. Trails-MD applies no biasing forces, but exploration mode redraws velocities
at every respawn, so a pair must never straddle a respawn boundary — the same constraint
that governs the time-lagged CV estimators:

```python
starts, ends = [], []
for iteration in iterations:
    cv = np.load(iteration / "cvs.npz")["cvs"]
    per_walker = len(cv) // n_walkers
    for w in range(n_walkers):
        seg = cv[w * per_walker:(w + 1) * per_walker]
        starts.append(seg[:-lag])
        ends.append(seg[lag:])
```

`lag` here only sets which pairs are formed; unlike an MSM, the fixed point is not a
function of it. Checking that the answer is flat across several lags is a cheap and
informative diagnostic.

### Convergence

`RiteWeightResult.weight_drift` records the per-iteration $L_1$ change in the weight
vector. A settled run shows it falling and then fluctuating about a small value.
`result.converged` applies a heuristic to that trace; it is reported rather than
enforced, so you can judge convergence instead of assuming it.

### Cost

Each iteration re-clusters every pooled point, so cost grows as
`n_iterations × n_points`. The nearest-centre assignment uses a KD-tree over the
(few) cluster centres, which is exact and turns an otherwise prohibitive
$O(N k)$ scan into roughly $O(N \log k)$ — the difference between feasible and not for
a campaign with $10^6$ pooled frames.

!!! note "Attribution"
    This is an independent implementation written from the published algorithm. The
    authors' reference code (`github.com/ZuckermanLab/rite_weight`) carries no licence
    statement and was deliberately not copied or vendored. If you use this feature,
    cite the PNAS paper.
