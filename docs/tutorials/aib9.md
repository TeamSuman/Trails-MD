# Tutorial: AIB9

AIB9 is a more stringent test than alanine dipeptide: the goal isn't only to
visit distinct conformational basins, but to determine whether sampled
trajectories connect the left- and right-handed helical states (L and R). A
low-dimensional coordinate can separate the two endpoints without proving
that a dynamical transition between them has actually been sampled — see
[Results in the paper](../results_in_paper.md#aib9-basin-discovery-is-not-a-transition-pathway)
for the full picture. This tutorial walks through the sampling-space options
on the path to that result.

## 1. Environment setup

```bash
conda env create -f env.yml
conda activate trails-md
cd examples/AIB9
```

All configs here use OpenMM with CUDA; switch `platform_name` to `CPU` if you
don't have a GPU available (slower).

## 2. Fixed torsional CVs

Start with a fixed, physically interpretable sampling space — the φ/ψ
dihedrals of residue 5:

```bash
trails-md --config config_fixed_phi_psi.yaml --check
trails-md --config config_fixed_phi_psi.yaml --iterations 50
```

As in the paper, a single torsional pair can rapidly cover its own local
plane while distal dihedrals remain weakly coupled — apparent local coverage
doesn't necessarily mean the full peptide has made a connected L-to-R
transition.

## 3. Learned CVs: TVAE

`config_phi_psi.yaml` learns a 2D TVAE space from φ/ψ-derived features;
`config_adaptive.yaml` learns one from pairwise distances instead:

```bash
trails-md --config config_phi_psi.yaml --iterations 50
trails-md --config config_adaptive.yaml --iterations 50
```

The TVAE model retrains periodically (`retrain_freq`) as more data
accumulates, and historical frames are reprojected into the updated latent
space each time it retrains.

## 4. Deep-TICA

For a nonlinear, dynamics-aware alternative, use Deep-TICA (requires
`pip install -e ".[deep-tica]"`):

```bash
trails-md --config config_deep_tica.yaml --iterations 50
```

## 5. Target-directed spawning

`config_target.yaml` learns a 2D TICA space and biases spawning toward a
specified target region (`search_mode: target`) rather than pure
exploration:

```bash
trails-md --config config_target.yaml --iterations 50
```

## 6. Check pathway connectivity, not just coverage

After a run, inspect `cvs.npz` to locate the L and R basins in your chosen
projection, then use `trails-md-path` to check whether the ancestry actually
connects them:

```bash
trails-md-path \
  --run-dir runs/aib9_target \
  --topology aib9_equilibrated.pdb \
  --start=<L-basin coordinates> \
  --end=<R-basin coordinates> \
  --output aib9_path.xtc
```

If no connected path exists between two well-sampled regions, that's the
signal that the campaign found both basins independently rather than a
transition between them — exactly the distinction the paper highlights for
AIB9.

## 7. Scale out on a cluster

`config_pbs.yaml` runs the same TICA-space campaign dispatched as PBS array
jobs instead of locally:

```bash
trails-md --config config_pbs.yaml --iterations 50
```

See [Execution](../execution.md) for SLURM/PBS configuration details.

## Next steps

- Read [Results in the paper](../results_in_paper.md) for how this connects
  to the manuscript's full AIB9 analysis.
- Read [Collective variables](../cv_methods.md) for the full list of
  available CV methods.
