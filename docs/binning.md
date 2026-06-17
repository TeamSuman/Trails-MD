# Adaptive binning

Bins stratify the CV space for the **density** and **weighted-ensemble** (`we`)
spawners. By default they are a **uniform grid** (`RegularBinner`): constant width
everywhere. That is suboptimal near barriers — a wide bin across a steep
free-energy region lets a walker slide back before it can reach the next bin
within the lag time, so weighted-ensemble flux across the barrier stalls; in flat
basins fine bins waste replicas.

`binning.scheme` selects a **landscape-adaptive** alternative, recomputed every
iteration. All schemes are opt-in; `uniform` is the default and exactly
reproduces the previous behaviour.

| `scheme` | What it does |
| --- | --- |
| `uniform` | Constant-width grid (default; `RegularBinner`). |
| `gradient` | **Equi-resistance** edges — boundaries at equal increments of `∫ exp(βF) dx ∝ ∫ 1/P dx`, so bins concentrate where the sampled density is low (barriers / steep regions) and widen in basins. |
| `mab` | Minimal-Adaptive-Binning style: uniform bins between the occupied extremes plus narrow "foothold" bins at the moving fronts. |
| `eigenvector` | Bin uniformly along the **leading (slowest) CV coordinate** only. For a learned CV / committor proxy this is automatically fine across the barrier and coarse in basins, and handles many CVs as a single 1-D coordinate. |

## Configuration

```yaml
binning:
  scheme: gradient     # uniform | gradient | mab | eigenvector
  n_fine: 100          # density-histogram resolution (gradient)
  smoothing: 3         # density smoothing window (gradient)

n_bins: [30, 30]       # number of bins per CV axis (target count for adaptive schemes)
```

## Notes

- **Weighted ensemble:** the `we` spawner carries per-frame statistical weights
  (not per-bin), so re-binning every iteration needs no weight remapping — the
  adaptive bins simply change which bin each frame falls in.
- **Resolution:** the configured `n_bins` is the per-axis target count; the global
  occupancy-driven resolution bump still applies and propagates to the adaptive
  binner.
- **Choosing a scheme:** `gradient` is the rigorous default for resolving barriers
  from the density alone; `eigenvector` is the cleanest choice with a learned CV
  or when the MSM is enabled (it bins along the slow coordinate); `mab` suits
  directed / front-pushing exploration.

## Extending

Register a new scheme by subclassing `AdaptiveBinner`
(`autosampler/binning/adaptive.py`) — implement `_axis_edges` (and optionally
`_coords`) — and adding it to `BinnerFactory`. It then works with both spawners
unchanged.
