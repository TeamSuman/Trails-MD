# Exploration vs. kinetics: choosing a mode

TRAILS-MD runs in one of two modes. They answer different questions, and picking
the right one for your problem matters more than any single algorithm choice.
Decide this first; everything else (CV space, spawner, engine, backend) is
secondary.

## The two modes at a glance

| | **Exploration mode** (default) | **Kinetics mode** |
|---|---|---|
| Question it answers | *What can this system do?* Which states exist, and is there a connected pathway to a target? | *How fast?* An unbiased rate (MFPT) between states. |
| Walker respawn | velocities **redrawn** (Maxwell–Boltzmann) — independent restarts | velocities **inherited** from the parent — continuous dynamics |
| Respects equilibrium weight? | No — that is why it crosses barriers fast | Yes — statistical weight conserved exactly |
| Gives a rate? | No | Yes |
| Typical spawners | `density`, `fps`, `lof`, `voronoi`, `msm` | `we` (weighted ensemble) |
| Enable with | *(nothing — this is the default)* | `spawn_scheme: we` + `inherit_velocities: true` |

## Exploration mode — discover pathways and states

This is the default and the right starting point for almost every project. Walkers
are respawned from under-sampled regions with fresh velocities, so the sampler
rapidly fans out across configuration space. Because it deliberately ignores the
equilibrium weight of trajectories, it reaches rare states far faster than plain
MD — the left-handed α basin of alanine dipeptide, the closed state of adenylate
kinase, and the cis state of proline are all found this way in hundreds of
nanoseconds where unbiased MD needs microseconds or never arrives.

Every walker records its parent, so once a target state is reached you can trace a
**lineage-connected pathway** back to the start — a continuous chain of
parent–child segments, not a scatter of disconnected snapshots. This is the main
output of exploration mode.

```yaml
spawning:
  spawn_scheme: density      # or fps / lof / voronoi / msm
  search_mode: explore
  # inherit_velocities defaults to false
```

What exploration mode does **not** give you is a rate. Velocity resampling
perturbs the dynamics, so the frequency of crossing events is not physically
meaningful. Do not read an MFPT off an exploration run.

## Kinetics mode — quantify a known pathway

When you need an unbiased rate, switch to kinetics mode. Walkers continue from
their parent's endpoint velocities, so weighted-ensemble split/merge resamples
*unperturbed* dynamics while conserving statistical weight exactly.

A rate needs **two** ingredients together, not just velocity inheritance:

1. `inherit_velocities: true` — continue the parent's dynamics (no fresh
   Maxwell–Boltzmann draw), so the resampled ensemble is unbiased dynamics.
2. `recycle_target` — a source→sink recycling box. A walker whose endpoint enters
   the box is terminated and its weight restarted at the basis (source) frame with
   fresh velocities. This drives a non-equilibrium steady state whose recycled
   weight per τ *is* the probability flux into the target, so
   **MFPT = 1 / flux** (the Hill relation — the same estimator WESTPA uses).

Inheritance alone gives you continuous dynamics but **no rate**: without
`recycle_target` no flux is booked and the MFPT is undefined.

```yaml
spawning:
  spawn_scheme: we
  we_target_per_bin: 4
  inherit_velocities: true              # (1) continue dynamics
  recycle_target: [[-2.5, -1.0], [2.0, 3.0]]   # (2) sink box: one [lo, hi] per CV dimension
  recycle_basis_index: 0                # source frame walkers restart from on recycling

engine:
  md_engine: openmm                     # kinetics mode is OpenMM-only
```

Requirements enforced at config load (clear error otherwise):

- `inherit_velocities` requires `spawn_scheme: we` **and** `md_engine: openmm`
  (only weighted ensemble keeps a set of continuable live walkers).
- `recycle_target` requires `spawn_scheme: we`, and must be bounded in **every** CV
  dimension — one `[lo, hi]` box per `n_bins` axis. An unbounded dimension would
  recycle walkers that never reached the target and bias the rate fast.

A runnable, self-contained (CPU-only) example ships at
`examples/alanine_dipeptide/config_kinetics.yaml` — a C7eq → αR rate on alanine
dipeptide.

### Reading the rate

The simplest way is the analysis CLI, which reports the MFPT and writes a
flux/running-MFPT convergence plot:

```bash
trails-md-analyze --run-dir runs/my_kinetics_run
```

```text
Weighted-ensemble kinetics  (Hill relation:  MFPT = tau / flux)
  MFPT estimate      : 0.352 ns
  tau (segment)      : 2 ps
  iterations         : 800 (412 with recycled flux)
  discard fraction   : 0.5 (leading transient dropped)
  flux plateau ratio : 0.98 (2nd half / 1st half of retained tail)
  status             : converged
  flux plot          : runs/my_kinetics_run/analysis/flux_convergence.png
```

τ (= `step * dt`) is read from the run log automatically; pass `--tau-ps` or
`--config` if the log is unavailable, and `--discard-fraction` to change the
transient cut. **Always check the `status` / plateau ratio** — a `NOT converged`
line (or a still-drifting flux plot) means the steady state has not been reached and
the run needs more iterations.

During a run, the current estimate is also logged each iteration
(`Kinetics: MFPT ~ … ns …`) so you can watch it settle.

The same number is available programmatically — `spawner.mfpt(tau_ps=step * dt)` or,
with diagnostics, `trails_md.spawners.we.steady_state_mfpt(flux_history, tau_ps)`.
Both discard the leading `discard_fraction` as the pre-steady-state transient
(reporting the un-discarded average is the most common way a WE rate comes out wrong)
and return `None` until some weight has been recycled.

An equivalent route to a rate is to build a **Markov state model** from the
trajectories of an *exploration* run — see [MSM & kinetic seeding](msm.md). Use the
MSM route when you already have exploration data; use weighted-ensemble kinetics
mode when you want to target the rate directly. Unlike the MSM route, the
steady-state flux estimate involves **no lag time**, so it is immune to the
lag/segment-length trap that biases short-segment MSMs.

## The intended workflow

1. **Explore** with a cheap spawner to discover the metastable states, a connected
   pathway, and a good coordinate.
2. If — and only if — you need a rate, **quantify** it along that coordinate with
   kinetics-mode weighted ensemble, or by seeding an MSM from the exploration data.

Both steps live in the same framework and the same configuration file: you change
the spawner and one flag, not the system, the engine, or the analysis. There is no
need to re-implement anything to go from *what happens* to *how fast*.
