# Trails-MD examples

Each example is a self-contained config plus the assets it needs. Paths inside a
config are resolved relative to that config file. Validate any example without
running MD using `--check`:

```bash
trails-md --config examples/<path>/config.yaml --check
```

## Start here (laptop-friendly, no GPU)

| Example | Demonstrates | Requirements |
| --- | --- | --- |
| [`alanine_dipeptide/config.yaml`](alanine_dipeptide/) | **Hello world** — fixed phi/psi CVs, density spawning, the full adaptive loop end-to-end | CPU only; vacuum Amber14 system, **no external force-field files**. Runs in minutes. |

The alanine-dipeptide system (`system.xml`) is built from the bundled 22-atom
`structure.pdb` (the canonical OpenMM test structure, MIT) by
`build_system.py` — rerun it to regenerate the asset. This is the recommended
first run:

```bash
trails-md --config examples/alanine_dipeptide/config.yaml --iterations 5
```

## AIB9 peptide (uses the bundled `aib9_system.xml`)

| Example | Demonstrates | Requirements |
| --- | --- | --- |
| `AIB9/config_fixed_phi_psi.yaml` | Fixed CVs on AIB9 | GPU recommended (solvated system) |
| `AIB9/config_phi_psi.yaml` | phi/psi projection | GPU recommended |
| `AIB9/config_adaptive.yaml` | Learned CV (TVAE) | GPU recommended |
| `AIB9/config_msm_vampnet.yaml` | VAMPNet CV + MSM convergence + MSM spawning + gradient binning | GPU recommended; `pip install '.[deep-tica]'` |
| `AIB9/config_msm_feature_selection.yaml` | VAMP-2 feature selection + MSM | GPU recommended |
| `AIB9/config_spib.yaml` | SPIB learned CV | GPU recommended |
| `AIB9/config_deep_tica.yaml` | deep-TICA learned CV | GPU recommended; `pip install '.[deep-tica]'` |
| `AIB9/config_we.yaml` | Weighted-ensemble (WE) spawner | GPU recommended |
| `AIB9/config_target.yaml` | Target-mode search toward a CV point | GPU recommended |
| `AIB9/config_pbs.yaml` | PBS/Torque HPC backend (array job per iteration) | A PBS cluster |
| `AIB9/config_tda_phi_sweep.yaml` | TDA phi sweep | Needs external Zenodo assets (see the config header) |

## Alanine dipeptide (GROMACS topology via OpenMM)

| Example | Demonstrates | Requirements |
| --- | --- | --- |
| `AlaD/config.yaml` | Fixed phi/psi, density spawner | A GROMACS install (its `share/gromacs/top` for the amber99sb.ff includes); set `engine.gromacs_include_dir`. GPU recommended. |
| `AlaD/config_voronoi.yaml` | Voronoi spawner | Same as above |

## Run scripts

- `run_local.sh` — run on a workstation (local backend).
- `slurm_submit.sh` / `pbs_submit.sh` — submit on an HPC cluster (see
  [docs/execution.md](../docs/execution.md)).

## Notebook

- `notebooks/adaptive_msm_tutorial.ipynb` — a rendered, GPU-free walkthrough of
  the components (feature selection → MSM → convergence → weighted ensemble →
  analysis) on synthetic data.

> Run outputs are written under each example's `outdir` (e.g. `runs/...`) and are
> not committed.
