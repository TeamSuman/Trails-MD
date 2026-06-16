# MSM & convergence

With `msm.enabled: true`, every iteration (subject to `cadence` and
`min_frames`) AutoSampler builds a **Markov State Model** from the cumulative
sampled data and uses it to decide when sampling is **complete**.

## What the MSM estimator does

`autosampler/msm/estimator.py` (`MSMEstimator`, built on `deeptime`) runs:

1. **Discretise** the CV/latent space into microstates (`cluster_method`:
   `kmeans` or `regspace`, `n_microstates`).
2. **Count** transitions at `lagtime`, restricting to the largest connected set.
3. **Estimate** the transition matrix — `mle` (maximum likelihood) or
   `bayesian` (posterior samples for statistical error bars).
4. **Analyse**: implied timescales, **VAMP-2** score, **PCCA+** metastable
   states (`n_metastable`), and the stationary distribution / free energy.
5. Optionally sweep `lagtimes` for an implied-timescale diagnostic.

Results are written to `iter_*/msm.npz` and checkpointed.

## Convergence criteria

A `ConvergenceMonitor` combines pluggable criteria; sampling stops when the
chosen combination holds for `convergence_patience` consecutive iterations.

| Criterion (`name`) | Triggers when… |
| --- | --- |
| `implied_timescales` | The slowest `n_timescales` implied timescales change by less than `tol` (relative). |
| `vamp2` | The VAMP-2 score change falls below `tol`. |
| `stationary_distribution` | The stationary distribution drift (L1/KL) falls below `tol`. |
| `statistical_error` | The Bayesian relative error on the slow timescales falls below `tol`. |

```yaml
msm:
  enabled: true
  lagtime: 10
  lagtimes: [1, 2, 5, 10, 20]
  n_microstates: 100
  estimator: bayesian
  n_metastable: 4
  convergence_mode: all          # all | any
  convergence_patience: 3
  convergence_criteria:
    - name: implied_timescales
      params: {tol: 0.1, n_timescales: 2}
    - name: vamp2
      params: {tol: 0.05}
    - name: statistical_error
      params: {tol: 0.2}
```

## MSM-guided spawning

Pair MSM convergence with `spawn_scheme: msm` to **drive** convergence: the
MSM least-counts spawner restarts walkers from microstates with the largest
statistical uncertainty, reducing the error on the slow processes fastest.

```yaml
spawning:
  spawn_scheme: msm
  voronoi_clusters: 100      # microstate count used by the spawner
```

## Practical notes

- Short walkers can produce **disconnected** counts early on; the estimator
  restricts to the largest connected set and early MSMs should be treated as
  diagnostics. Use `min_frames` to delay the first MSM.
- Build cost scales with frames; raise `cadence` to estimate the MSM less often.
- All MSM behaviour is **off by default**, so non-MSM runs are unaffected.
