# Trails-MD

**Trails-MD** is a modular framework for **adaptive molecular dynamics
campaigns**. It runs many short MD walkers, projects saved frames into a
fixed or machine-learned collective-variable (CV) space, restarts walkers
from informative regions, and repeats the cycle.

## Key features

- **Engine-agnostic walkers.** OpenMM, GROMACS, and Amber share the same
  adaptive loop.
- **Fixed or learned sampling spaces.** User-defined physical CVs, PCA, TICA,
  TVAE, Deep-TICA, VAMPnet, SPIB, and the supervised Deep-LDA / Deep-TDA
  coordinates — all swappable with one keyword. See
  [Collective variables](cv_methods.md).
- **VAMP-2 feature selection.** Reduce thousands of candidate distances to the
  few dozen that actually carry the slow dynamics.
  See [Feature selection](feature_selection.md).
- **Six interchangeable spawning policies.** Density, Voronoi,
  local-outlier-factor, farthest-point, **weight-conserving weighted ensemble**,
  and **MSM-guided** (least-counts × slow-mode leverage × uncertainty).
  See [Concepts](concepts.md).
- **Landscape-adaptive binning.** Uniform, density-gradient, minimal adaptive
  binning (MAB), and MSM-eigenvector schemes. See [Adaptive binning](binning.md).
- **In-loop Markov State Models.** Estimate an MSM *inside* the adaptive loop and
  stop the campaign on **kinetic** convergence — implied timescales, VAMP-2,
  stationary distribution, transition matrix, or Bayesian statistical error —
  rather than on bin occupancy alone. See [MSM & kinetic seeding](msm.md).
- **Lineage-aware exploration.** Every spawned frame stores its parent-child
  ancestry, so connected transition pathways can be reconstructed from
  otherwise disjoint exploration stages.
- **Restartable campaigns.** Per-iteration checkpoints capture the adaptive
  model, feature history, sampling state, and walker coordinates; deterministic
  seeding means a resumed run reproduces an uninterrupted one.
- **HPC scalability.** Run on a multi-GPU workstation or dispatch walkers as
  **SLURM** / **PBS** array jobs (`execution.backend`), with submit-retry against
  queue limits, configurable GPU request directives, and per-walker GPU isolation
  checks. See [Execution](execution.md).

## The adaptive loop

```text
  run short MD walkers (local / SLURM / PBS)
              |
  extract features / project to CV space
              |
  train or update the CV if space_mode is a learned
  method (PCA / TICA / TVAE / Deep-TICA)
              |
  spawn new walkers (density / Voronoi / LOF / FPS / WE / MSM)
              |
  iteration budget reached, or bin occupancy
  plateaued? --- yes ---> stop
              |
              no
              |
        (back to the top)
```

Sampling continues until either the configured iteration budget is reached
or grid/Voronoi bin occupancy plateaus (see [Concepts](concepts.md)). After a
campaign, representative structures can seed longer unbiased production runs
for post-hoc Markov State Model (MSM) construction — see
[MSM & kinetic seeding](msm.md).

## Where to go next

- New here? Start with the **[Quickstart](quickstart.md)**.
- Want the full picture? Read **[Concepts](concepts.md)**.
- Configuring a run? See the **[Configuration reference](configuration.md)**.
- Running on a cluster? See **[Execution](execution.md)**.
- Worked end-to-end examples: **[Alanine dipeptide](tutorials/alad.md)** and
  **[AIB9](tutorials/aib9.md)**.
- Curious what Trails-MD found in the paper? See
  **[Results in the paper](results_in_paper.md)**.
