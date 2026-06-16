# Collective variables

AutoSampler can sample in **fixed** physical CVs or **learn** CVs on the fly.
The available learned methods live in a single registry
(`autosampler/spaces/registry.py`), which also tracks each method's backend and
whether it is available in your environment.

## Available methods

| `space_mode` | Backend | Time-lagged | Notes |
| --- | --- | --- | --- |
| `pca` | scikit-learn | no | Linear baseline. |
| `tica` | deeptime | yes | Linear, dynamics-aware. |
| `tvae` | deeptime + torch | yes | Time-lagged variational autoencoder. |
| `vampnet` | deeptime + torch | yes | Deep CVs via the VAMP-2 variational principle. |
| `spib` | torch (built-in) | yes | State Predictive Information Bottleneck (Wang & Tiwary, 2021). |
| `deep-tica` | mlcolvar + lightning | yes | Deep nonlinear TICA (optional extra). |
| `deep-lda` | mlcolvar + lightning | no | Supervised; needs per-frame state labels (optional extra). |

`fixed` mode uses a user `project_file` exposing
`extract_cvs(trajectories, top_file, conf_file) -> ndarray`.

## Choosing a method

- **Start simple:** `tica` (fast, robust, interpretable) or `pca`.
- **Nonlinear / deep CVs:** `vampnet` or `spib` are strong defaults and need no
  extra packages beyond torch.
- **Supervised separation of known states:** `deep-lda`.

## Configuring

```yaml
space_mode: vampnet
adaptive_feature_type: distances      # distances | fitted_coords | phi_psi
retrain_freq: 5                       # retrain the CV every 5 iterations
adaptive_model:
  lagtime: 5
  latent_dim: 2
  epochs: 50
  encoder_hidden_dims: [64, 32]
  spib_n_states: 10                   # used when space_mode: spib
  spib_beta: 0.001
```

## Adaptive retraining (VAMP-2 driven)

A learned CV can go stale as new regions are discovered. By default it retrains
on a fixed schedule (`retrain_freq`). Set `retrain_policy: vamp_adaptive` to
retrain **only when the CV's VAMP-2 score on fresh data drops** below its
post-training reference — coupling retraining to sampling progress.

```yaml
retrain_policy: vamp_adaptive   # fixed | vamp_adaptive
vamp_retrain_tol: 0.1           # relative VAMP-2 drop that triggers a retrain
retrain_min_interval: 1         # don't retrain more often than this
retrain_max_interval: 20        # force a refresh at least this often (optional)
```

The controller's reference score is checkpointed, so `--resume` continues the
same policy.

## Availability checks

If a method's backend is missing, AutoSampler raises an actionable error, e.g.:

```text
CV method 'deep-tica' requires missing package(s): mlcolvar, lightning.
Install via: pip install "autosampler[deep-tica]".
```

Programmatically:

```python
from autosampler.spaces.registry import is_available, adaptive_modes
adaptive_modes()          # ('pca','tica','tvae','vampnet','spib','deep-tica','deep-lda')
is_available("vampnet")   # True / False
```

## Adding a new CV method

Register a `CVMethod` in `autosampler/spaces/registry.py` and add a branch in
`AdaptiveSpaceModel.fit` / `.project`. The rest of the framework (training
cadence, MSM, spawning) works unchanged.
