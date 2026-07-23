# MSM & kinetic seeding

Trails-MD separates **adaptive exploration** from **kinetic estimation**. Walkers are
short, and their velocities are redrawn from a Maxwell–Boltzmann (MB) distribution at
each spawn point.

## What velocity resampling does and does not do

!!! info "MB resampling is **not** a bias — but it does cap your lag time"
    For a separable Hamiltonian $H(x,p)=K(p)+U(x)$ the Boltzmann density factorizes, so
    momenta are statistically **independent** of positions at equilibrium. Drawing
    $p \sim \mathrm{MB}(T)$ at any position $x$ therefore reproduces *exactly* the
    equilibrium conditional density $\rho(p\mid x)$. A segment launched from
    $(x,\,p\sim\mathrm{MB})$ is a legitimate realization of the equilibrium dynamics
    conditioned on $x$.

    Likewise, **adaptively choosing where to start walkers does not bias the MSM.** Each
    row of the transition matrix is estimated from the counts *leaving* that state, which
    are unbiased estimates of $p(j \mid i, \tau)$ no matter how often you chose to start
    there. Adaptive selection changes *where you collect statistics*, not the transition
    probabilities. This is exactly why adaptive-sampling → MSM is a standard, rigorous
    workflow.

    **What resampling does cost is trajectory continuity.** It severs phase-space
    continuity at every parent→child boundary, so each walker segment must be treated as
    an independent trajectory (Trails-MD does this correctly and never stitches walkers
    together). The consequence is a hard cap:

    $$\tau \le L \qquad (L = \text{walker segment length})$$

    You cannot count a transition at a lag longer than the trajectory carrying it.

!!! warning "The trap: convergence in *iteration* is not convergence in *lag*"
    Markovianity also needs $\tau \gg 1/\gamma$ (the momentum relaxation time, ~1 ps for
    `friction = 1 ps⁻¹`), so the usable window is $1/\gamma \ll \tau \le L$. If the implied
    timescales have **not plateaued by $\tau = L$**, the MSM systematically *underestimates*
    the slow timescales — and the failure is silent, because you cannot push $\tau$ past $L$
    to test it. The convergence monitor watches the ITS across *iterations*: as data
    accumulate the estimate stops moving and looks "converged" while still being wrong.

    Trails-MD therefore **refuses to certify convergence when `msm.lagtime` exceeds 1/5 of
    the shortest walker segment**, and warns. If you need a longer lag, increase
    `spawning.step` so that segments get longer.

For quantitative kinetics on slow processes, prefer the two-stage strategy below (or seed
longer unbiased runs and reweight with TRAM/dTRAM).

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
