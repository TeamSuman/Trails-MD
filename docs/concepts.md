# Concepts

## Two modes: exploration vs. kinetics

TRAILS-MD runs in one of two modes, and choosing the right one matters more than any
other setting:

- **Exploration mode** (default) — walkers respawn with *fresh* velocities, fanning out
  across configuration space to discover states and pathways fast. The data is excellent
  for coverage and lineage but **biased for rates**.
- **Kinetics mode** — walkers *inherit* their parent's velocities (`inherit_velocities:
  true`) so weighted ensemble resamples unperturbed dynamics; paired with source→sink
  recycling (`recycle_target`) it yields an unbiased mean-first-passage time via the Hill
  relation (`MFPT = 1/flux`).

Decide this first. See [Exploration vs. kinetics](modes.md) for the full comparison.

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
| `we` | Weighted-ensemble split/merge resampling with **exact weight conservation** (split → `w/c`, merge sums, `sum(w) = 1` every iteration). CPU is allocated **bin-balanced, never weight-proportional**, and the binner is re-fitted to the **live** walker ensemble each iteration (not the cumulative history) so a frontier bin always holds a walker to replicate. `we_target_per_bin` sets walkers per bin. In the default exploration mode velocities are resampled, so a WE run is great for coverage but not a rate; for an unbiased rate/MFPT, use **kinetics mode** (`inherit_velocities: true` + source→sink `recycle_target`) — see [Exploration vs. kinetics](modes.md). |
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

Convergence can also be judged on the *kinetic model* when `msm.enabled` is set: the
`ConvergenceMonitor` stops on implied-timescale, VAMP-2, stationary-distribution, and
statistical-error criteria — see [MSM & kinetic seeding](msm.md) and
[Analysis](analysis.md).

After a campaign, representative structures can seed longer production runs
for post-hoc MSM construction — see [MSM & kinetic seeding](msm.md). For a rate targeted
directly during sampling, use kinetics mode instead — see [Exploration vs. kinetics](modes.md).

## Execution

Walkers are dispatched by an **execution backend**: `local` (multi-GPU
workstation) or `slurm` / `pbs` (HPC array jobs). The choice is purely a
config setting and does not affect the science. See [Execution](execution.md).

## Reproducibility & provenance

- Global deterministic seeding (`random_seed`).
- Per-iteration checkpoints (`checkpoint_freq`) with `--resume`.
- Every frame carries lineage (`iteration:walker:frame` + parent), enabling
  connected-path reconstruction with `trails-md-path`.
