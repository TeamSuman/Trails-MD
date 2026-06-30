# AutoSampler

AutoSampler is a Python framework for adaptive molecular dynamics campaigns.
It runs many short MD walkers, projects saved frames into a collective-variable
or learned latent space, chooses informative restart frames, and repeats the
cycle — continuing until a **Markov State Model (MSM)** built on the sampled
data has **converged**.

The code is meant for method development and practical sampling workflows where
you need to change engines, CVs, spawning policies, or analysis criteria without
rewriting the whole pipeline.

## What's new

- **MSM-convergence engine** — build an MSM each iteration (deeptime) and stop
  automatically on implied-timescale / VAMP-2 convergence *and* a flux-weighted
  **transition-matrix** statistical-error gate (`spawn_scheme: msm`, `msm.enabled`).
- **Uncertainty-guided spawning** — seed walkers by **uncertainty × leverage ×
  flux** to drive the MSM toward convergence fastest (`msm.spawn_uncertainty`).
- **Landscape-adaptive binning** — place bins finer across barriers and coarser
  in basins (`binning.scheme: gradient | mab | eigenvector`), recomputed each
  iteration; defaults to the uniform grid.
- **More learned CVs** — VAMPNet and SPIB alongside TICA / TVAE / PCA / deep-TICA.
- **VAMP-2 feature selection** — optionally select and adaptively update the
  input features that best resolve the slow dynamics (`feature_selection.enabled`).
- **HPC scalability** — run on a multi-GPU workstation or dispatch walkers as
  **SLURM** / **PBS** array jobs (`execution.backend`).

Get started in one command — `autosampler-init` writes an annotated input file
covering every method, feature, and hyperparameter (see
[`docs/input_file.md`](docs/input_file.md)). A runnable
[notebook tutorial](examples/notebooks/adaptive_msm_tutorial.ipynb) with rendered
plots walks through the whole workflow.

See **[`docs/`](docs/index.md)** (full documentation & tutorials) and
**[`CHANGELOG.md`](CHANGELOG.md)**. Build the docs site with
`pip install mkdocs-material && mkdocs serve`.

## Motivation

Long molecular transitions are often missed by a single continuous trajectory.
Adaptive sampling attacks this by running short trajectory batches and restarting
new walkers from frames that look under-sampled, unusual, far apart, or close to
a target region.

AutoSampler focuses on three practical requirements:

- **Modularity:** OpenMM, GROMACS, and Amber-style engines share the same
  adaptive loop.
- **Interpretable sampling spaces:** fixed physical CVs, custom project-file
  projections, PCA, TICA, TVAE, and Deep-TICA can be swapped at configuration
  time.
- **Reproducible provenance:** every iteration stores projections, trajectory
  paths, spawn indices, frame lineage, and checkpoints so runs can be resumed
  and post-processed into connected paths.

The AIB9 and alanine dipeptide examples illustrate the intended scientific use:
coverage in a projected space is not automatically a transition pathway.
AutoSampler records lineage so basin discovery can be distinguished from a
connected mechanistic path.

## Repository Structure

```text
autosampler/
  cli.py                    Command-line entry point for adaptive runs
  config.py                 Pydantic configuration schema
  core.py                   Main adaptive sampling controller
  engines/                  OpenMM, GROMACS, and Amber backends
  spaces/                   Feature extraction and adaptive latent spaces
  spawners/                 Density, Voronoi, LOF, and farthest-point spawning
  binning/                  Regular-grid and Voronoi binning utilities
  checkpoints/              Checkpoint save/load logic
  paths.py, path_cli.py     Lineage-aware path reconstruction
  logs.py, log_cli.py       Run-log generation utilities

examples/
  AlaD/                     Alanine dipeptide fixed phi/psi examples
  AIB9/                     AIB9 fixed and learned CV examples

```

## Installation

Create the conda environment from `env.yml`. It installs AutoSampler in editable
mode and uses the project metadata in `pyproject.toml`.

```bash
conda env create -f env.yml
conda activate autosampler
```

Optional Deep-TICA extras:

```bash
python -m pip install -e ".[deep-tica]"
```

External engine executables must also be installed separately if you use those
backends:

- GROMACS executable for `engine.md_engine: gromacs`
- Amber/pmemd executable for `engine.md_engine: amber`

## Quick Start

Validate a configuration before running MD:

```bash
autosampler --config examples/AlaD/config.yaml --check
```

Run an adaptive campaign:

```bash
autosampler --config examples/AlaD/config.yaml --iterations 20
```

Resume from the latest checkpoint:

```bash
autosampler --config examples/AlaD/config.yaml --resume --iterations 20
```

Resume from a specific checkpoint:

```bash
autosampler --config examples/AlaD/config.yaml --resume 10 --iterations 20
```

Generate a post-hoc exploration log for a completed run:

```bash
autosampler-log \
  --run-dir examples/AlaD/runs/alad_phi_psi_density \
  --config examples/AlaD/config.yaml
```

Reconstruct a connected lineage path between two CV-space points:

```bash
autosampler-path \
  --run-dir examples/AlaD/runs/alad_phi_psi_density \
  --topology examples/AlaD/start.gro \
  --start=-1.05,-0.70 \
  --end=1.05,0.70 \
  --output alad_path.xtc
```

For batch path extraction, use `--pairs-file` and `--output-dir`.

## CLI Help

Main adaptive runner:

```text
usage: autosampler [-h] [--config CONFIG] [--iterations ITERATIONS]
                   [--resume [RESUME]] [--check]
                   [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]

options:
  --config CONFIG       YAML config path. Relative paths inside it are resolved
                        from this file.
  --iterations N        Number of adaptive iterations to run.
  --resume [RESUME]     Resume from latest checkpoint, or from checkpoints/iter_N.
  --check               Validate inputs and executables, then exit before MD.
  --log-level LEVEL     Python logging verbosity.
```

Connected path post-processing:

```text
usage: autosampler-path --run-dir RUN_DIR --topology TOPOLOGY
                        [--start START] [--end END] [--output OUTPUT]
                        [--pairs-file PAIRS_FILE] [--output-dir OUTPUT_DIR]
                        [--metadata METADATA] [--checkpoint CHECKPOINT]
```

Exploration log generation:

```text
usage: autosampler-log --run-dir RUN_DIR [--config CONFIG]
                       [--output OUTPUT] [--n-bins N_BINS]
                       [--min-values MIN_VALUES] [--max-values MAX_VALUES]
                       [--append]
```

## Configuration Overview

A minimal OpenMM configuration looks like:

```yaml
system:
    conf_file: structure.pdb
    top_file: structure.pdb
    project_file: project_cvs.py

engine:
    md_engine: openmm
    platform_name: CUDA

spawning:
    spawn_scheme: density
    walker: 10
    step: 10000
    stride: 100
    max_workers: 2

space_mode: fixed
n_bins: [30, 30]
min_values: [-3.14159, -3.14159]
max_values: [3.14159, 3.14159]
outdir: runs/my_sampling_run
```

When `space_mode: fixed`, `system.project_file` should define:

```python
def extract_cvs(trajectories, top_file, conf_file):
    ...
    return cvs  # shape: (n_frames, n_cvs)
```

See `examples/AlaD/project_phi_psi.py` and
`examples/AIB9/project_phi_psi.py` for concrete examples.

## Spawning Strategies

AutoSampler currently supports:

- `density`: regular-grid low-population bin selection.
- `voronoi`: KMeans-backed Voronoi cells with exact clipped polygon areas.
- `lof`: local-outlier-factor based frame selection.
- `fps`: farthest-point sampling for geometric spread.
- `msm`: MSM least-counts — restart from sparsely-sampled microstates to
  directly reduce the statistical error of the Markov State Model.

Voronoi note: `voronoi_clusters` controls the restart-selection partition.
The regular `n_bins` grid is still used for run-log coverage diagnostics.

## Collective-Variable (CV) Methods

The sampling space is chosen with `space_mode`. Beyond fixed user CVs, several
learned CV methods are available through a single registry
(`autosampler/spaces/registry.py`), so new methods can be added in one place:

| `space_mode` | Method | Backend | Notes |
|--------------|--------|---------|-------|
| `fixed`      | User CVs via a project file | — | e.g. AlaD `phi/psi` |
| `pca`        | Principal component analysis | scikit-learn | linear baseline |
| `tica`       | Time-lagged ICA | deeptime | linear, dynamics-aware |
| `tvae`       | Time-lagged VAE | deeptime + torch | nonlinear bottleneck |
| `vampnet`    | VAMPNet | deeptime + torch | trained with the VAMP-2 score |
| `spib`       | State Predictive Information Bottleneck | built-in (torch) | Wang & Tiwary 2021 |
| `deep-tica`  | Deep (nonlinear) TICA | mlcolvar (optional) | `pip install "autosampler[deep-tica]"` |
| `deep-lda`   | Deep LDA (supervised) | mlcolvar (optional) | needs state labels |

`vampnet` and `spib` work out of the box (only deeptime/torch). Optional methods
raise a clear, actionable error if their backend is missing.

## MSM-Based Convergence

With `msm.enabled: true`, AutoSampler builds a Markov State Model over the CV
space each iteration (clustering → transition counts → MLE/Bayesian MSM →
implied timescales, VAMP-2 score, PCCA+ metastable states) and stops sampling
once the MSM has **converged**. Convergence is decided by a composable
`ConvergenceMonitor` with pluggable criteria — implied-timescale stability,
VAMP-2 plateau, stationary-distribution drift, and Bayesian statistical error —
combined with `all`/`any` and a patience window. Per-iteration results are
written to `iter_*/msm.npz`. See `examples/AIB9/config_msm_vampnet.yaml` for a
complete MSM-driven adaptive-sampling configuration.

## Typical Workflow

1. **Define the scientific question.**
   Decide whether you need basin discovery, endpoint structures, or a connected
   transition path. These are different claims.

2. **Choose a sampling space.**
   Start with physical CVs when possible, for example AlaD `phi/psi`. Use
   learned spaces when physical CVs are unclear, then validate the results with
   interpretable observables.

3. **Write or select a YAML config.**
   Start from one of the YAML files under `examples/`. Keep path values relative
   to the config file when possible.

4. **Preflight the run.**

    ```bash
    autosampler --config config.yaml --check
    ```

5. **Run adaptive sampling.**

    ```bash
    autosampler --config config.yaml --iterations 100
    ```

6. **Monitor `output.log`.**
   The log records per-iteration timings, successful walkers, cumulative
   frames, diagnostic occupied bins, exploration fraction, spawn indices, and
   checkpoint paths.

7. **Resume if needed.**

    ```bash
    autosampler --config config.yaml --resume --iterations 50
    ```

8. **Analyze coverage and endpoints.**
   Use `cvs.npz`, `output.log`, and example notebooks/scripts to inspect the
   sampled CV space.

9. **Check lineage before claiming a pathway.**
   Use `autosampler-path` to reconstruct connected trajectories. Endpoint
   proximity alone does not prove that a transition was sampled.

10. **Archive configs, logs, checkpoints, and analysis.**
    A reproducible campaign should retain the YAML config, project CV file,
    `output.log`, checkpoints, and any scripts used to classify states.

## Examples

Alanine dipeptide:

```bash
cd examples/AlaD
autosampler --config config.yaml --check
autosampler --config config.yaml --iterations 2
```

AlaD Voronoi smoke test:

```bash
cd examples/AlaD
autosampler --config config_voronoi.yaml --iterations 1
```

AIB9 fixed phi/psi CV:

```bash
cd examples/AIB9
autosampler --config config_fixed_phi_psi.yaml --check
```

## Current Status

This is an active research codebase. The examples are useful starting points,
but production scientific claims should be made only after checking CV
definitions, sampling bounds, lineage connectivity, and system-specific
validation.

## How to cite

If you use AutoSampler in your research, please cite it. Citation metadata is in
[`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button
from it). A DOI and the accompanying publication will be added on release.
