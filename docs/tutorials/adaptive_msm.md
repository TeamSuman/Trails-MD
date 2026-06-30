# Tutorial: adaptive sampling to MSM convergence

This walkthrough runs AutoSampler on the **AIB9** peptide, learning a VAMPNet CV
on the fly, building an MSM each iteration, and stopping automatically when the
MSM converges. It uses the shipped example
`examples/AIB9/config_msm_vampnet.yaml`.

## 1. Environment

```bash
conda env create -f env.yml
conda activate autosampler
pip install -e ".[deep-tica]"
```

## 2. Inspect the configuration

Key settings (see the file for the full version):

```yaml
space_mode: vampnet            # learn a deep CV
adaptive_feature_type: distances
retrain_freq: 5                # retrain the CV every 5 iterations

spawning:
  spawn_scheme: msm            # MSM least-counts seeding
  walker: 10
  step: 5000
  stride: 50

msm:
  enabled: true
  lagtime: 10
  estimator: bayesian
  n_metastable: 4
  convergence_mode: all
  convergence_patience: 3
  stable_clustering: true        # comparable microstate IDs across iterations
  spawn_uncertainty: true        # uncertainty × leverage × flux seeding
  convergence_criteria:
    - {name: implied_timescales, params: {tol: 0.1, n_timescales: 2}}
    - {name: vamp2, params: {tol: 0.05}}
    - {name: transition_matrix, params: {tol: 0.2, min_flux: 1.0e-4}}
```

With `convergence_mode: all`, the run stops only when the slow kinetics have
stabilised **and** the flux-weighted statistical uncertainty of the transition
matrix `T_ij` has fallen below `tol` — so a plateau in timescales alone will not
end the run while important transitions are still poorly estimated. The
`spawn_scheme: msm` spawner above actively drives that uncertainty down by
seeding walkers from microstates scored by **uncertainty × leverage × flux**
(see [MSM & convergence](../msm.md)).

### Adaptive binning (optional)

The density / WE spawners bin the CV space on a uniform grid by default. Switch
to landscape-adaptive bins — finer across barriers, coarser in basins,
recomputed each iteration — with a `binning` block (see
[Adaptive binning](../binning.md)):

```yaml
binning:
  scheme: gradient     # uniform | gradient | mab | eigenvector
```

Optionally enable VAMP-2 feature selection (see
[Feature selection](../feature_selection.md)):

```yaml
feature_selection:
  enabled: true
  method: greedy_vamp
  lagtime: 10
  cadence: 5
```

## 3. Preflight

```bash
autosampler --config examples/AIB9/config_msm_vampnet.yaml --check
```

This validates inputs, the engine, and settings without running MD.

## 4. Run

```bash
autosampler --config examples/AIB9/config_msm_vampnet.yaml --iterations 200 --log-level INFO
```

Each iteration prints a summary banner and appends a row to
`runs/adaptive_msm_vampnet/output.log`. The run **stops early** when the
ConvergenceMonitor reports convergence:

```text
Converged: implied timescales and VAMP-2 score satisfied for 3 iterations.
```

## 5. What gets written

```text
runs/adaptive_msm_vampnet/
├── output.log                 # per-iteration metrics
├── iter_0/ … iter_N/
│   ├── iteration_*_*.xtc       # walker trajectories
│   ├── cvs.npz                 # CV projections
│   ├── features.npz            # input features (save_features: true)
│   └── msm.npz                 # MSM diagnostics (timescales, VAMP-2, π, PCCA+)
└── checkpoints/iter_*/         # resumable state
```

## 6. Resume if needed

```bash
autosampler --config examples/AIB9/config_msm_vampnet.yaml --resume --iterations 100
```

## 7. Analyse

```bash
autosampler-log --run-dir runs/adaptive_msm_vampnet
```

Load the MSM diagnostics for plotting:

```python
import numpy as np
data = np.load("runs/adaptive_msm_vampnet/iter_40/msm.npz", allow_pickle=True)
print(sorted(data.files))           # timescales, vamp2_score, stationary_distribution, ...
```

## 8. Run it on a cluster

Switch only the `execution` section to scale out — see
[Execution](../execution.md). No other changes are needed.
