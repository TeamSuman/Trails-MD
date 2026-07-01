# Trails-MD

**Trails-MD** is a modular framework for **adaptive molecular dynamics
campaigns**. It runs many short MD walkers, projects saved frames into a
fixed or machine-learned collective-variable (CV) space, restarts walkers
from informative regions, and repeats the cycle.

## Key features

- **Engine-agnostic walkers.** OpenMM, GROMACS, and Amber share the same
  adaptive loop.
- **Fixed or learned sampling spaces.** User-defined physical CVs, PCA, TICA,
  TVAE, and Deep-TICA, swappable at configuration time.
- **Interchangeable spawning policies.** Density, Voronoi,
  local-outlier-factor, and farthest-point selection.
- **Lineage-aware exploration.** Every spawned frame stores its parent-child
  ancestry, so connected transition pathways can be reconstructed from
  otherwise disjoint exploration stages.
- **Restartable campaigns.** Per-iteration checkpoints capture the adaptive
  model, feature history, sampling state, and walker coordinates.
- **HPC scalability.** Run on a multi-GPU workstation or dispatch walkers as
  **SLURM** / **PBS** array jobs (`execution.backend`).

## The adaptive loop

```text
  run short MD walkers (local / SLURM / PBS)
              |
  extract features / project to CV space
              |
  train or update the CV if space_mode is a learned
  method (PCA / TICA / TVAE / Deep-TICA)
              |
  spawn new walkers (density / Voronoi / LOF / FPS)
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
