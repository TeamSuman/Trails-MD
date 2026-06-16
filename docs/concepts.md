# Concepts

## Walkers, iterations, and spawning

Each **iteration** runs a batch of short MD **walkers**. Saved frames are
projected into a CV space; a **spawner** then chooses which frames to restart
the next iteration's walkers from. Spawners:

| `spawn_scheme` | Strategy |
| --- | --- |
| `density` | Restart from under-populated regions of a regular grid. |
| `voronoi` | Restart from sparse Voronoi (k-means) cells. |
| `lof` | Restart from statistical outliers (local outlier factor). |
| `fps` | Farthest-point sampling for maximal coverage. |
| `msm` | **MSM least-counts**: restart from microstates with the largest statistical uncertainty (drives MSM convergence). |

## CV spaces

A run uses either:

- **Fixed CVs** (`space_mode: fixed`) — a user `project_file` returning physical
  CVs (dihedrals, distances, …), or
- **Learned CVs** (`space_mode: tica | tvae | vampnet | spib | deep-tica | pca`)
  — trained on the fly from input features and periodically retrained
  (`retrain_freq`). See [Collective variables](cv_methods.md).

## Input features

Learned CVs are trained on **input features** extracted from the trajectories:
pairwise `distances`, `fitted_coords`, or system-specific dihedrals, restricted
by the `feature_selection` atom mask. Optionally, a **VAMP-2 feature selector**
chooses and adaptively updates the best subset — see
[Feature selection](feature_selection.md).

## MSM and convergence

When `msm.enabled` is set, each iteration discretises the CV space into
microstates, estimates a transition matrix at a lag time, and computes implied
timescales, the VAMP-2 score, and PCCA+ metastable states. A
**ConvergenceMonitor** combines pluggable criteria (timescale stability, VAMP-2
plateau, stationary-distribution drift, statistical error) to decide when
sampling is complete. See [MSM & convergence](msm.md).

## Execution

Walkers are dispatched by an **execution backend**: `local` (multi-GPU
workstation) or `slurm` / `pbs` (HPC array jobs). The choice is purely a config
setting and does not affect the science. See [Execution](execution.md).

## Reproducibility & provenance

- Global deterministic seeding (`random_seed`).
- Per-iteration checkpoints (`checkpoint_freq`) with `--resume`.
- Every frame carries lineage (`iteration:walker:frame` + parent), enabling
  connected-path reconstruction with `autosampler-path`.
