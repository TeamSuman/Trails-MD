# Quickstart

## Install

```bash
conda env create -f env.yml
conda activate autosampler
pip install -e ".[deep-tica]"     # optional deep-TICA / deep-LDA backends
```

For a lightweight environment (MSM, CV registry, feature selection, and tests
without the heavy MD backends):

```bash
pip install numpy scipy scikit-learn pydantic pyyaml deeptime pytest torch
```

## Validate a configuration

`--check` runs all preflight checks (files, executables, settings) and exits
without launching MD:

```bash
autosampler --config examples/AlaD/config.yaml --check
```

## Run

```bash
# Fixed phi/psi CVs, density spawning, 20 iterations
autosampler --config examples/AlaD/config.yaml --iterations 20

# Adaptive VAMPNet CV + MSM convergence (stops automatically)
autosampler --config examples/AIB9/config_msm_vampnet.yaml --iterations 200
```

When `msm.enabled` is set, the run stops as soon as the MSM converges, even if
the iteration budget is not exhausted.

## Resume

Every iteration is checkpointed. Resume from the latest (or a specific) one:

```bash
autosampler --config examples/AIB9/config_msm_vampnet.yaml --resume --iterations 50
autosampler --config examples/AIB9/config_msm_vampnet.yaml --resume 12 --iterations 50
```

## Inspect results

```bash
# Per-iteration coverage / timing log
autosampler-log --run-dir runs/adaptive_msm_vampnet

# MSM analysis report (VAMP-2 / timescales / free energy / network)
autosampler-analyze --run-dir runs/adaptive_msm_vampnet

# Reconstruct a connected path between two CV points
autosampler-path \
  --run-dir runs/alad_phi_psi_density \
  --topology examples/AlaD/start.gro \
  --start=-1.05,-0.70 --end=1.05,0.70 \
  --output alad_path.xtc
```

Each `iter_*/` directory holds the trajectories, `cvs.npz`, optional
`features.npz`, and (when enabled) `msm.npz`. `output.log` is a tab-separated
per-iteration record.
