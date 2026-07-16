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
*unperturbed* dynamics while conserving statistical weight exactly. The weights
then support an unbiased mean-first-passage-time estimate.

```yaml
spawning:
  spawn_scheme: we
  we_target_per_bin: 4
  inherit_velocities: true   # continue dynamics; required for a valid rate

engine:
  md_engine: openmm          # kinetics mode is OpenMM-only
```

Kinetics mode requires `spawn_scheme: we` (only weighted ensemble maintains a set
of continuable live walkers) and the OpenMM engine; the configuration is rejected
with a clear message otherwise.

An equivalent route to a rate is to build a **Markov state model** from the
trajectories of an *exploration* run — see [MSM & kinetic seeding](msm.md). Use the
MSM route when you already have exploration data; use weighted-ensemble kinetics
mode when you want to target the rate directly.

## The intended workflow

1. **Explore** with a cheap spawner to discover the metastable states, a connected
   pathway, and a good coordinate.
2. If — and only if — you need a rate, **quantify** it along that coordinate with
   kinetics-mode weighted ensemble, or by seeding an MSM from the exploration data.

Both steps live in the same framework and the same configuration file: you change
the spawner and one flag, not the system, the engine, or the analysis. There is no
need to re-implement anything to go from *what happens* to *how fast*.
