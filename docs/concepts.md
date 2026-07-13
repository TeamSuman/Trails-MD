# Concepts

## Walkers, iterations, and spawning

Each **iteration** runs a batch of short MD **walkers**. Saved frames are
projected into a CV space; a **spawner** then chooses which frames to restart
the next iteration's walkers from. Spawners:

| `spawn_scheme` | Strategy |
| --- | --- |
| `density` | Restart from under-populated regions of a regular grid. |
| `voronoi` | Restart from sparse Voronoi (k-means) cells; scales better than a grid in higher-dimensional CV spaces. |
| `lof` | Restart from statistical outliers (local outlier factor). |
| `fps` | Farthest-point sampling for maximal coverage. |
| `we` | Weighted-ensemble split/merge resampling, **conserving total statistical weight** (`we_target_per_bin`). Note that the default MB velocity resampling breaks exact trajectory continuity — disable it, or use a dedicated WE package, if you need formal WE rate guarantees. |
| `msm` | MSM-guided spawning (needs `msm.enabled`): least-counts × slow-mode leverage × outflow uncertainty (`msm.spawn_alpha`, `spawn_leverage`, `spawn_uncertainty`). Targets the states whose sampling most improves the *kinetic model*, not merely the geometrically sparse ones. |

Any spawner can also be pointed toward a target region of the CV space
(`search_mode: target`), balancing exploration with progress toward the
target.

## CV spaces

A run uses either:

- **Fixed CVs** (`space_mode: fixed`) — a user `project_file` returning physical
  CVs (dihedrals, distances, …), or
- **Learned CVs** (`space_mode: pca | tica | tvae | deep-tica | vampnet | spib | deep-lda`) — trained on
  the fly from input features and periodically retrained (`retrain_freq`).
  See [Collective variables](cv_methods.md).

## Input features

Learned CVs are trained on **input features** extracted from the
trajectories: pairwise `distances`, `fitted_coords`, or system-specific
dihedrals, restricted by the `feature_selection` atom mask.

## Convergence

Sampling proceeds for the configured iteration budget, or stops early when
grid/Voronoi bin occupancy plateaus (`resolution_check_patience`,
`convergence_patience` in `spawning`). Because a retrained learned CV space
can rotate, shift, or scale, bin boundaries are recalculated in the newly
projected space whenever the model retrains.

After a campaign, representative structures can seed longer production runs
for post-hoc MSM construction — see [MSM & kinetic seeding](msm.md).

## Execution

Walkers are dispatched by an **execution backend**: `local` (multi-GPU
workstation) or `slurm` / `pbs` (HPC array jobs). The choice is purely a
config setting and does not affect the science. See [Execution](execution.md).

## Reproducibility & provenance

- Global deterministic seeding (`random_seed`).
- Per-iteration checkpoints (`checkpoint_freq`) with `--resume`.
- Every frame carries lineage (`iteration:walker:frame` + parent), enabling
  connected-path reconstruction with `trails-md-path`.
