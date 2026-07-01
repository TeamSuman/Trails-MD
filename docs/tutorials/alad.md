# Tutorial: Alanine dipeptide

Alanine dipeptide is the simplest useful test case for Trails-MD: its
relevant low-dimensional collective variables (the backbone dihedrals φ and
ψ) are already known, so the sampled coverage can be interpreted directly in
terms of well-established physical coordinates.

## 1. Environment setup

```bash
conda env create -f env.yml
conda activate trails-md
cd examples/AlaD
```

This example uses OpenMM and needs no extra optional dependencies.

## 2. Inspect the configuration

`config.yaml` defines a fixed `φ`/`ψ` sampling space with density-based
spawning:

```yaml
space_mode: fixed
n_bins: [30, 30]
min_values: [-3.14159, -3.14159]
max_values: [3.14159, 3.14159]
```

`project_phi_psi.py` implements `extract_cvs(...)`, returning the two
backbone dihedrals for each frame.

## 3. Preflight and run

```bash
trails-md --config config.yaml --check
trails-md --config config.yaml --iterations 20
```

Output (trajectories, `cvs.npz`, checkpoints, `output.log`) is written under
`runs/alad_phi_psi_density/`.

## 4. Compare against Voronoi spawning

`config_voronoi.yaml` runs the same system with `spawn_scheme: voronoi`
instead of `density`:

```bash
trails-md --config config_voronoi.yaml --iterations 20
```

## 5. Inspect coverage

```bash
trails-md-log --run-dir runs/alad_phi_psi_density --config config.yaml
trails-md-log --run-dir runs/alad_phi_psi_voro --config config_voronoi.yaml
```

Each log records per-iteration occupied bins out of the configured 30×30
grid. Both spawning schemes should show density- and Voronoi-based coverage
climbing steadily and reaching a broadly similar fraction of the
Ramachandran plane, consistent with the comparison in the paper (Section
IV.A).

## 6. Reconstruct a connected path

Use `trails-md-path` to trace a connected trajectory between two points in
the (φ, ψ) plane, e.g. from the `C7eq` basin toward `αR`:

```bash
trails-md-path \
  --run-dir runs/alad_phi_psi_density \
  --topology start.gro \
  --start=-1.05,-0.70 \
  --end=1.05,0.70 \
  --output alad_path.xtc
```

This reconstructs a connected ancestry-based trajectory (not just two
endpoint frames), which is the distinction Trails-MD's lineage tracking is
designed to make: coverage of a region isn't the same as having sampled a
connected transition through it.

## Next steps

- Try a learned CV space on a harder system — see the
  [AIB9 tutorial](aib9.md).
- Read [Concepts](../concepts.md) for the full picture of spawners, CV
  spaces, and convergence.
