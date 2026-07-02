# Trails-MD

Trails-MD is a Python framework for adaptive molecular dynamics campaigns.
It runs many short MD walkers, projects saved frames into a collective-variable
or learned latent space, chooses informative restart frames, and repeats the
cycle — continuing until a sufficiently converged space is reached.

The code is meant for method development and practical sampling workflows where
you need to change engines, CVs, spawning policies, or analysis criteria without
rewriting the whole pipeline.

## Key features

- **Engine-agnostic walkers** — OpenMM, GROMACS, and Amber share the same
  adaptive loop.
- **Fixed or learned sampling spaces** — user-defined physical CVs, PCA, TICA,
  TVAE, and Deep-TICA, swappable at configuration time.
- **Interchangeable spawning policies** — density, Voronoi, local-outlier-factor,
  and farthest-point selection.
- **Lineage-aware exploration** — every spawned frame stores its parent–child
  ancestry, so connected transition pathways can be reconstructed from otherwise
  disjoint exploration stages.
- **Restartable campaigns** — per-iteration checkpoints capture the adaptive
  model, feature history, sampling state, and walker coordinates.
- **HPC scalability** — run on a multi-GPU workstation or dispatch walkers as
  **SLURM** / **PBS** array jobs (`execution.backend`).

A runnable [notebook tutorial](examples/notebooks/adaptive_msm_tutorial.ipynb)
with rendered plots walks through the whole workflow, and an annotated input
file covering the available methods and hyperparameters is documented in
[`docs/input_file.md`](docs/input_file.md).

See **[`docs/`](docs/index.md)** (full documentation & tutorials) and
**[`CHANGELOG.md`](CHANGELOG.md)**. Build the docs site with
`pip install mkdocs-material && mkdocs serve`.

## Motivation

Long molecular transitions are often missed by a single continuous trajectory.
Adaptive sampling attacks this by running short trajectory batches and restarting
new walkers from frames that look under-sampled, unusual, far apart, or close to
a target region.

Trails-MD focuses on three practical requirements:

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
Trails-MD records lineage so basin discovery can be distinguished from a
connected mechanistic path.

## Repository Structure

```text
trails_md/
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

Create the conda environment from `env.yml`. It installs Trails-MD in editable
mode and uses the project metadata in `pyproject.toml`.

```bash
conda env create -f env.yml
conda activate trails-md
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
trails-md --config examples/AlaD/config.yaml --check
```

Run an adaptive campaign:

```bash
trails-md --config examples/AlaD/config.yaml --iterations 20
```

Resume from the latest checkpoint:

```bash
trails-md --config examples/AlaD/config.yaml --resume --iterations 20
```

Resume from a specific checkpoint:

```bash
trails-md --config examples/AlaD/config.yaml --resume 10 --iterations 20
```

Generate a post-hoc exploration log for a completed run:

```bash
trails-md-log \
  --run-dir examples/AlaD/runs/alad_phi_psi_density \
  --config examples/AlaD/config.yaml
```

Reconstruct a connected lineage path between two CV-space points:

```bash
trails-md-path \
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
usage: trails-md [-h] [--config CONFIG] [--iterations ITERATIONS]
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
usage: trails-md-path --run-dir RUN_DIR --topology TOPOLOGY
                        [--start START] [--end END] [--output OUTPUT]
                        [--pairs-file PAIRS_FILE] [--output-dir OUTPUT_DIR]
                        [--metadata METADATA] [--checkpoint CHECKPOINT]
```

Exploration log generation:

```text
usage: trails-md-log --run-dir RUN_DIR [--config CONFIG]
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

Trails-MD currently supports:

- `density`: regular-grid low-population bin selection.
- `voronoi`: KMeans-backed Voronoi cells with exact clipped polygon areas.
- `lof`: local-outlier-factor based frame selection.
- `fps`: farthest-point sampling for geometric spread.

Voronoi note: `voronoi_clusters` controls the restart-selection partition.
The regular `n_bins` grid is still used for run-log coverage diagnostics.

## Collective-Variable (CV) Methods

The sampling space is chosen with `space_mode`. Beyond fixed user CVs, several
learned CV methods are available through a single registry
(`trails_md/spaces/registry.py`), so new methods can be added in one place:

| `space_mode` | Method                       | Backend             | Notes                                |
| ------------ | ---------------------------- | ------------------- | ------------------------------------ |
| `fixed`      | User CVs via a project file  | —                   | e.g. AlaD `phi/psi`                  |
| `pca`        | Principal component analysis | scikit-learn        | linear baseline                      |
| `tica`       | Time-lagged ICA              | deeptime            | linear, dynamics-aware               |
| `tvae`       | Time-lagged VAE              | deeptime + torch    | nonlinear bottleneck                 |
| `deep-tica`  | Deep (nonlinear) TICA        | mlcolvar (optional) | `pip install "trails-md[deep-tica]"` |

When a model is retrained, the full feature history is reprojected into the
updated latent space before spawning, so selection always reflects the current
coordinates. Optional methods raise a clear, actionable error if their backend
is missing.

## Post-Processing and Kinetic Seeding

Trails-MD separates adaptive exploration from kinetic estimation. Walkers are
short and their velocities are redrawn from a Maxwell–Boltzmann distribution at
each spawn point, so the adaptive trajectories are intended for exploration
rather than as an unbiased kinetic ensemble. After a campaign, the explored
space can be discretized and representative structures selected to seed longer,
unbiased production trajectories. Those production runs are the appropriate
input for Markov State Model (MSM) construction and related kinetic analyses.

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
    trails-md --config config.yaml --check
    ```

5. **Run adaptive sampling.**

    ```bash
    trails-md --config config.yaml --iterations 100
    ```

6. **Monitor `output.log`.**
   The log records per-iteration timings, successful walkers, cumulative
   frames, diagnostic occupied bins, exploration fraction, spawn indices, and
   checkpoint paths.

7. **Resume if needed.**

    ```bash
    trails-md --config config.yaml --resume --iterations 50
    ```

8. **Analyze coverage and endpoints.**
   Use `cvs.npz`, `output.log`, and example notebooks/scripts to inspect the
   sampled CV space.

9. **Check lineage before claiming a pathway.**
   Use `trails-md-path` to reconstruct connected trajectories. Endpoint
   proximity alone does not prove that a transition was sampled.

10. **Archive configs, logs, checkpoints, and analysis.**
    A reproducible campaign should retain the YAML config, project CV file,
    `output.log`, checkpoints, and any scripts used to classify states.

## Examples

Alanine dipeptide:

```bash
cd examples/AlaD
trails-md --config config.yaml --check
trails-md --config config.yaml --iterations 2
```

AlaD Voronoi smoke test:

```bash
cd examples/AlaD
trails-md --config config_voronoi.yaml --iterations 1
```

AIB9 fixed phi/psi CV:

```bash
cd examples/AIB9
trails-md --config config_fixed_phi_psi.yaml --check
```

## Current Status

This is an active research codebase. The examples are useful starting points,
but production scientific claims should be made only after checking CV
definitions, sampling bounds, lineage connectivity, and system-specific
validation.

## License

Trails-MD is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
It's free to use for research, academic, educational, and other noncommercial
purposes; commercial use requires a separate license from the copyright
holder. See the [`LICENSE`](LICENSE) file for the full terms.

<!-- ## How to cite

If you use Trails-MD in your research, please cite it. Citation metadata is in
[`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button
from it). A DOI and the accompanying publication will be added on release.

Unhide this section once the accompanying paper is published. -->

