# AutoSampler

**AutoSampler** is a modular framework for **autonomous adaptive molecular
dynamics sampling**. It runs many short MD walkers, projects frames into a
fixed or machine-learned collective-variable (CV) space, restarts walkers from
informative regions, and repeats — continuing until a **Markov State Model
(MSM)** built on the sampled data has **converged**.

## Why AutoSampler

- **MSM-convergence driven.** Sampling proceeds until implied timescales and the
  VAMP-2 score plateau, *and* the flux-weighted statistical error on the
  transition matrix falls below threshold — not just until bins fill up.
- **Landscape-adaptive binning.** Optionally place bins finer across barriers and
  coarser in basins (`gradient` / `mab` / `eigenvector`), recomputed each
  iteration, instead of a uniform grid.
- **Learned or fixed CVs.** Use physical CVs (dihedrals, distances, …) or learn
  them on the fly: TICA, TVAE, **VAMPNet**, **SPIB**, deep-TICA, PCA.
- **VAMP-2 feature optimisation.** Optionally select and adaptively update the
  input features that best resolve the slow dynamics.
- **Runs anywhere.** A multi-GPU workstation (local backend) or CPU/GPU HPC
  clusters via **SLURM** or **PBS** array jobs, with automatic resubmission.
- **Reproducible.** Deterministic seeding, full checkpoint/restart, and
  lineage-aware path reconstruction.

## The adaptive loop

```text
        ┌─────────────────────────────────────────────────────────┐
        │  run short MD walkers (local / SLURM / PBS)              │
        │            │                                            │
        │   extract features ──► (VAMP-2 feature selection) ──┐   │
        │            │                                        │   │
        │   train / update CV (TICA / VAMPNet / SPIB / …)  ◄──┘   │
        │            │                                            │
        │   build MSM (clusters → T(τ) → ITS / VAMP-2 / PCCA+)    │
        │            │                                            │
        │   converged? ──yes──► stop                              │
        │            │no                                          │
        │   spawn new walkers (MSM least-counts / density / …)    │
        └────────────┴────────────────────────────────────────────┘
```

## Where to go next

- New here? Start with the **[Quickstart](quickstart.md)**.
- Want the full picture? Read **[Concepts](concepts.md)**.
- Configuring a run? See the **[Configuration reference](configuration.md)**.
- Tuning where bins go? See **[Adaptive binning](binning.md)**.
- Running on a cluster? See **[Execution](execution.md)**.
- A worked end-to-end example: the **[adaptive-MSM tutorial](tutorials/adaptive_msm.md)**.
