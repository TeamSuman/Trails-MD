# The input file

Everything about a run — the system, MD engine, sampling method, CV space,
feature selection, MSM convergence, and where jobs execute — is described by a
single YAML **input file**. There is no code to write for a standard run.

## Get a starter file

```bash
trails-md-init                 # writes ./config.yaml (annotated template)
trails-md-init -o my_run.yaml  # custom path
```

Then edit it and validate before running:

```bash
trails-md --config config.yaml --check     # checks files, engine, settings
trails-md --config config.yaml --iterations 200
```

The starter file is also at `examples/template.yaml`, and worked examples live
under `examples/AlaD/` and `examples/AIB9/`.

## Structure

The file has one block per concern. Only `system` (and `project_file` for
`space_mode: fixed`) is mandatory; everything else has sensible defaults, and
the advanced blocks (`feature_selection`, `msm`, `execution`, retrain policy)
are **opt-in**.

| Block | Selects |
| --- | --- |
| `system` | structure, topology, and the atom mask for features |
| `engine` | MD backend (OpenMM/GROMACS/Amber) and thermodynamics |
| `spawning` | how the next walkers are chosen, and walker/step counts |
| `space_mode` + `adaptive_model` | the CV method and its hyperparameters |
| `feature_selection` | VAMP-2 selection/optimisation of input features |
| `msm` | MSM estimation and the convergence criteria that stop the run |
| `execution` | where walkers run: workstation, SLURM, or PBS |
| run-level keys | `outdir`, `random_seed`, `checkpoint_freq`, … |

## Choosing methods (the knobs that matter most)

**Sampling method** — `spawning.spawn_scheme`:
`density` · `voronoi` · `lof` · `fps` · `msm` (least-counts, drives MSM
convergence) · `we` (weighted ensemble).

**CV method** — `space_mode`:
`fixed` (your `project_file`) · `pca` · `tica` · `tvae` · `vampnet` · `spib` ·
`deep-tica` · `deep-lda`. Hyperparameters live in `adaptive_model`
(`lagtime`, `latent_dim`, `epochs`, `encoder_hidden_dims`, …). See
[Collective variables](cv_methods.md).

**Features** — `adaptive_feature_type` (`distances`/`fitted_coords`/`phi_psi`)
restricted by `system.feature_selection`. Turn on
[`feature_selection`](feature_selection.md) to let VAMP-2 pick the best subset
and/or feature type automatically.

**Convergence** — set `msm.enabled: true` to build an MSM each iteration and
stop when `msm.convergence_criteria` are satisfied (implied timescales, VAMP-2,
statistical error). See [MSM & convergence](msm.md).

**Where it runs** — `execution.backend`: `local` (multi-GPU workstation) or
`slurm` / `pbs` (HPC array jobs). See [Execution](execution.md).

## Annotated template

```yaml
--8<-- "examples/template.yaml"
```

!!! note
    The snippet above is the exact file `trails-md-init` writes. Every field
    is documented inline; the [Configuration reference](configuration.md) lists
    all keys, defaults, and allowed values in table form.

## How settings flow

`trails-md --config config.yaml` loads the YAML, validates it against the
schema (`trails_md/config.py`), resolves relative paths, and runs the adaptive
loop. Invalid values (e.g. an unknown `space_mode` or `spawn_scheme`) are
rejected immediately with a clear message, before any MD is launched.
```
