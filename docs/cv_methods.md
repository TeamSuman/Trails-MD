# Collective variables

> Primary references for TICA, TVAE, and Deep-TICA are collected on the
> [References](references.md) page.

Trails-MD can sample in **fixed** physical CVs or **learn** CVs on the fly.
The available learned methods live in a single registry
(`trails_md/spaces/registry.py`), which also tracks each method's backend and
whether it is available in your environment.

All eight methods below are implemented behind the same interface and are exercised
end-to-end in the test suite and in the alanine-dipeptide benchmark campaign. They
differ in *what they optimize*, which is the thing to reason about when choosing:
variance (PCA), slow modes (TICA, Deep-TICA, VAMPnet), time-lagged reconstruction
(TVAE), predictive information (SPIB), or endpoint discrimination (Deep-LDA).

## Available methods

| `space_mode` | Method                        | Optimizes | Backend |
| ------------- | ----------------------------- | --------- | ------- |
| `fixed`       | User CVs via a project file   | — (you choose)          | — |
| `pca`         | Principal component analysis  | variance                | scikit-learn |
| `tica`        | Time-lagged ICA               | autocorrelation (slow modes) | deeptime |
| `tvae`        | Time-lagged VAE                | time-lagged reconstruction (nonlinear) | deeptime + torch |
| `deep-tica`   | Deep (nonlinear) TICA          | slow modes (nonlinear)  | mlcolvar + lightning |
| `vampnet`     | VAMPNet                        | VAMP-2 score (soft metastable states) | deeptime + torch |
| `spib`        | State predictive information bottleneck | predictive information | torch |
| `deep-lda`    | Deep LDA (supervised)          | endpoint discrimination | mlcolvar + lightning |

!!! note "One caveat worth knowing"
    `spib` runs a **single-pass** information-bottleneck projection; it does *not*
    perform the iterative self-consistent state refinement of the original method.
    Treat its states as exploratory rather than converged metastable assignments.

`deep-tica` and `deep-lda` need the optional extra:
`pip install "trails-md[deep-tica]"`. `deep-lda` additionally requires per-frame
state labels.

`fixed` mode uses a user `project_file` exposing
`extract_cvs(trajectories, top_file, conf_file) -> ndarray`.

## Choosing a method

- **Start simple:** `tica` (fast, robust, interpretable) or `pca`.
- **Nonlinear CVs:** `tvae` or `deep-tica` when a good linear projection isn't
  enough to separate conformations that overlap in physical coordinates.
- **Kinetically meaningful states:** `vampnet` (soft metastable assignment) or `spib`.
- **You know the endpoints:** `deep-lda` — but note that endpoint
  *separation* is not the same as having sampled a *pathway* between them; check the
  lineage (see [Concepts](concepts.md)).
- **Pairs with kinetics mode:** a learned slow coordinate (`tica`/`deep-tica`) makes a
  natural progress coordinate for a weighted-ensemble rate — see
  [Exploration vs. kinetics](modes.md).
- Whatever you pick, **validate against an interpretable observable** for your system.

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
