# MSM & kinetic seeding

Trails-MD separates **adaptive exploration** from **kinetic estimation**.
Walkers are short, and their velocities are redrawn from a Maxwell-Boltzmann
distribution at each spawn point, so adaptive trajectories are designed for
efficient exploration of a CV space, not as an unbiased kinetic ensemble on
their own.

## The two-stage strategy

1. **Adaptive exploration.** Run a Trails-MD campaign (see
   [Concepts](concepts.md)) to discover conformational space and identify
   representative structures across the explored region.
2. **Kinetic seeding.** Use representative structures selected from the
   adaptive campaign to seed longer, unbiased production trajectories.
3. **Post-hoc MSM construction.** Build a Markov State Model from the
   production trajectories using standard external tools (e.g. `deeptime`) —
   clustering, transition-matrix estimation, implied-timescale analysis, and
   coarse-graining into metastable states.

This two-stage approach is useful because the adaptive stage identifies
representative starting structures across the explored space, while the
production stage generates trajectories that are directly suitable for MSM
estimation — avoiding the bias that would come from building a kinetic model
directly on short, velocity-randomized adaptive walkers.

## What Trails-MD provides for this workflow

- **Coverage diagnostics** (`trails-md-log`) to identify well- and
  under-sampled regions of the CV space at the end of a campaign.
- **Lineage tracking** (`trails-md-path`) to reconstruct connected
  parent-child trajectories, useful for selecting seeding structures along a
  hypothesized transition path.
- **Checkpointed campaign state**, so representative-structure selection can
  be revisited without rerunning the adaptive stage.

## Example: chignolin folding

The paper demonstrates this workflow on chignolin (CLN025) folding in
explicit water: an adaptive Trails-MD campaign explores the CV space,
representative spawn points are selected from the discretized explored
space, and long production trajectories seeded from those points are used to
build a two-state coarse-grained MSM with implied-timescale validation. See
[Results in the paper](results_in_paper.md#msm-construction-from-trails-md-seeded-trajectories).
