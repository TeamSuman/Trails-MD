# Feature selection (VAMP-2)

The quality of a learned CV — and the MSM built on it — is bounded by the
**input features** fed to it. AutoSampler can **select and adaptively update**
those features automatically using the **VAMP-2 score**, a variational measure
of how much slow kinetic variance a feature set captures (Wu & Noé 2017;
Scherer et al. 2019). Higher VAMP-2 = better features.

## How input features are produced

For learned CVs, features are extracted from each trajectory according to:

- `adaptive_feature_type`: `distances` (pairwise distances), `fitted_coords`
  (RMSD-aligned Cartesian coordinates), or `phi_psi` (system dihedrals);
- `feature_selection`: the MDAnalysis atom mask restricting which atoms
  contribute.

Without VAMP-2 selection these features are fixed for the whole run. With it,
AutoSampler keeps the **subset of feature columns that best resolves the slow
dynamics**, and refreshes that subset as more of the landscape is explored.

## Enabling it

```yaml
feature_selection:
  enabled: true
  method: greedy_vamp     # greedy forward selection (or `all` to keep everything)
  lagtime: 10             # lag time for VAMP-2 scoring
  cadence: 5              # re-select every 5 iterations (adaptive update)
  max_features: 50        # optional cap on the number of selected columns
  min_gain: 1.0e-4        # stop adding columns when the VAMP-2 gain is tiny
```

The selected columns are applied consistently at CV training, projection, and
historical re-projection, and are saved in the checkpoint so `--resume` keeps
the same selection.

## The optimisation protocol

`method: greedy_vamp` runs **greedy forward selection**: starting from no
features, it repeatedly adds the column whose inclusion most increases the
VAMP-2 score, stopping once no column improves the score by more than
`min_gain` (or `max_features` is reached). Because VAMP-2 is monotonic in the
number of features, the `min_gain` threshold is what yields a *parsimonious*
feature set that retains essentially all of the kinetic variance.

## Using the API directly

```python
import numpy as np
from autosampler.spaces.feature_selection import (
    vamp2_score, rank_candidates, greedy_vamp_selection, FeatureSelector,
)

# trajs: list of (n_frames, n_features) arrays, one per walker
score = vamp2_score(trajs, lagtime=10)

# Compare candidate feature sets:
rank_candidates({"distances": d_trajs, "dihedrals": phi_trajs}, lagtime=10)

# Optimise the column subset:
cols = greedy_vamp_selection(trajs, lagtime=10, max_groups=20)

# Or via the orchestrator used by the loop:
sel = FeatureSelector(lagtime=10, method="greedy_vamp").select(trajs)
sel.columns, sel.score
```

## Choosing among feature *types*

Beyond selecting columns within one feature type, AutoSampler can rank whole
**feature types** by VAMP-2 and use the best one. List the candidates and the
loop extracts each, ranks them, and switches to the winner (re-running column
selection when the type changes):

```yaml
feature_selection:
  enabled: true
  candidate_feature_types: [distances, fitted_coords]   # subset of:
                                                        # distances | fitted_coords | phi_psi
  cadence: 5            # re-rank types every 5 iterations
```

When `candidate_feature_types` is empty (default) the top-level
`adaptive_feature_type` is used. The chosen type is checkpointed for resume.
