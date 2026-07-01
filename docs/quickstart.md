# Quickstart

## Install

```bash
conda env create -f env.yml
conda activate trails-md
pip install -e ".[deep-tica]"     # optional deep-TICA / deep-LDA backends
```

For a lightweight environment (CV registry and tests without the heavy MD
backends):

```bash
pip install numpy scipy scikit-learn pydantic pyyaml deeptime pytest torch
```

## Validate a configuration

`--check` runs all preflight checks (files, executables, settings) and exits
without launching MD:

```bash
trails-md --config examples/AlaD/config.yaml --check
```

## Run

```bash
# Fixed phi/psi CVs, density spawning, 20 iterations
trails-md --config examples/AlaD/config.yaml --iterations 20

# Learned TICA CV on AIB9
trails-md --config examples/AIB9/config_target.yaml --iterations 50
```

## Resume

Every iteration is checkpointed. Resume from the latest (or a specific) one:

```bash
trails-md --config examples/AIB9/config_target.yaml --resume --iterations 50
trails-md --config examples/AIB9/config_target.yaml --resume 12 --iterations 50
```

## Inspect results

```bash
# Per-iteration coverage / timing log
trails-md-log --run-dir runs/aib9_target

# Reconstruct a connected path between two CV points
trails-md-path \
  --run-dir runs/alad_phi_psi_density \
  --topology examples/AlaD/start.gro \
  --start=-1.05,-0.70 --end=1.05,0.70 \
  --output alad_path.xtc
```

Each `iter_*/` directory holds the trajectories, `cvs.npz`, and optional
`features.npz`. `output.log` is a tab-separated per-iteration record.
