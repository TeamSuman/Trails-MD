# Quickstart

## Install

```bash
conda env create -f env.yml
conda activate trails-md
pip install -e ".[deep-tica]"     # optional deep-TICA / deep-LDA backends
```

For a lightweight environment (CV registry and tests without the full MD
stack) install the core dependencies and the package itself:

```bash
pip install numpy scipy scikit-learn pydantic pyyaml deeptime MDAnalysis shapely torch pytest
pip install -e .          # so `import trails_md` and the `trails-md` CLI work
```

## Validate a configuration

`--check` runs all preflight checks (files, executables, settings) and exits
without launching MD. Start with the self-contained CPU hello-world (no GPU,
no external force-field files):

```bash
trails-md --config examples/alanine_dipeptide/config.yaml --check
```

## Run

```bash
# Self-contained hello-world: fixed phi/psi CVs, density spawning, CPU
trails-md --config examples/alanine_dipeptide/config.yaml --iterations 20

# Learned TICA CV on AIB9
trails-md --config examples/AIB9/config_target.yaml --iterations 50
```

The `examples/AlaD/` configs use a GROMACS topology, so they additionally need
GROMACS installed and `engine.gromacs_include_dir` set — see
[the alanine-dipeptide tutorial](tutorials/alad.md).

## Resume

Every iteration is checkpointed. Resume from the latest (or a specific) one:

```bash
trails-md --config examples/AIB9/config_target.yaml --resume --iterations 50
trails-md --config examples/AIB9/config_target.yaml --resume 12 --iterations 50
```

## Inspect results

Output lands next to the config (`outdir` is resolved relative to the config file),
so the AIB9 run above writes to `examples/AIB9/runs/aib9_target`:

```bash
# Per-iteration coverage / timing log
trails-md-log --run-dir examples/AIB9/runs/aib9_target

# Reconstruct a connected path between two CV points (from the hello-world run;
# phi/psi in radians — adjust to basins your run sampled)
trails-md-path \
  --run-dir examples/alanine_dipeptide/runs/alanine_dipeptide_hello \
  --topology examples/alanine_dipeptide/structure.pdb \
  --start=-1.4,2.6 --end=-1.4,-0.7 \
  --output path.xtc
```

Each `iter_*/` directory holds the trajectories, `cvs.npz`, and optional
`features.npz`. `output.log` is a tab-separated per-iteration record.
