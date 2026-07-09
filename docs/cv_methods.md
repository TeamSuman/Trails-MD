# Collective variables

> Primary references for TICA, TVAE, and Deep-TICA are collected on the
> [References](references.md) page.

Trails-MD can sample in **fixed** physical CVs or **learn** CVs on the fly.
The available learned methods live in a single registry
(`trails_md/spaces/registry.py`), which also tracks each method's backend and
whether it is available in your environment.

> **Scope note.** `fixed`, `pca`, `tica`, `tvae`, and `deep-tica` are the
> methods exercised in the manuscript. `vampnet`, `spib`, and `deep-lda` are
> available through the same interface but are **experimental/beta**: validate
> them against interpretable observables for your system before drawing
> conclusions.

## Available methods

| `space_mode` | Method                        | Backend               | Notes |
| ------------- | ----------------------------- | ---------------------- | ----- |
| `fixed`       | User CVs via a project file   | —                       | e.g. dihedrals, distances. |
| `pca`         | Principal component analysis  | scikit-learn            | Linear baseline. |
| `tica`        | Time-lagged ICA               | deeptime                | Linear, dynamics-aware. |
| `tvae`        | Time-lagged VAE                | deeptime + torch        | Nonlinear bottleneck. |
| `deep-tica`   | Deep (nonlinear) TICA          | mlcolvar + lightning     | `pip install "trails-md[deep-tica]"`. |
| `vampnet`     | VAMPNet (deep VAMP-2 CVs)      | deeptime + torch        | **Experimental** (not in the manuscript). |
| `spib`        | State Predictive Info Bottleneck | torch                | **Experimental**; no self-consistent state refinement (see below). |
| `deep-lda`    | Deep LDA (supervised)          | mlcolvar + lightning     | **Experimental**; needs per-frame state labels. |

The last three modes share the same interface but are **beyond the current
manuscript scope** — the paper describes them as extension points, not validated
methods. `spib` runs a single-pass information-bottleneck projection without the
iterative self-consistent state refinement of the original method; treat its
states as exploratory, not converged metastable assignments. Validate any of the
three against interpretable observables for your system before drawing
conclusions.

`fixed` mode uses a user `project_file` exposing
`extract_cvs(trajectories, top_file, conf_file) -> ndarray`.

## Choosing a method

- **Start simple:** `tica` (fast, robust, interpretable) or `pca`.
- **Nonlinear CVs:** `tvae` or `deep-tica` when a good linear projection isn't
  enough to separate conformations that overlap in physical coordinates.

## Configuring

```yaml
space_mode: tica
adaptive_feature_type: distances      # distances | fitted_coords | phi_psi
retrain_freq: 5                       # retrain the CV every 5 iterations
adaptive_model:
  lagtime: 5
  latent_dim: 2
  epochs: 50
  encoder_hidden_dims: [64, 32]
```

When a model is retrained, the full accumulated feature history is
reprojected into the updated latent space before spawning, so selection
always reflects the current coordinates.

## Availability checks

If a method's backend is missing, Trails-MD raises an actionable error, e.g.:

```text
CV method 'deep-tica' requires missing package(s): mlcolvar, lightning.
Install via: pip install "trails-md[deep-tica]".
```

Programmatically:

```python
from trails_md.spaces.registry import is_available
is_available("tica")   # True / False
```

## Adding a new CV method

Register a `CVMethod` in `trails_md/spaces/registry.py` and add a branch in
`AdaptiveSpaceModel.fit` / `.project`. The rest of the framework (training
cadence, spawning) works unchanged.
