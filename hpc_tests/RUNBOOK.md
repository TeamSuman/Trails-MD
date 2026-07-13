# Trails-MD test runbook

An ordered procedure for an operator (or an automated test runner)
to (1) validate **all major features** without a cluster, (2) reproduce the known
correctness fixes, and (3) run and triage the **SLURM/PBS** feature matrix on a
cluster. Every step writes a structured JSON report so a failure can be localized
by `code` using [`DEBUGGING.md`](DEBUGGING.md).

The three test layers, fastest first:

| Layer | Command | Needs a cluster? | What it proves |
| --- | --- | --- | --- |
| Unit + regression | `python -m pytest -q` | no | logic incl. the A1/A2/A3 correctness fixes |
| **Local feature matrix** | `python hpc_tests/run_local_matrix.py` | no | every major feature runs end-to-end on the local backend |
| Scheduler matrix | `sbatch hpc_tests/slurm/run_features.sbatch` (or PBS) | yes | the same features under real SLURM/PBS array dispatch |

---

## 1. Local feature matrix (do this first — no cluster needed)

```bash
# See what will run vs SKIP on this machine (based on installed backends).
python hpc_tests/run_local_matrix.py --list

# Run everything runnable (a few minutes on the tiny alanine-dipeptide workloads).
python hpc_tests/run_local_matrix.py            # -> results/local/
```

It runs each feature end-to-end (`--check` → short `--iterations` → validate) on
the **local** execution backend and prints a PASS/SKIP/FAIL table. Coverage:

- **Engines:** OpenMM (`openmm_*`), GROMACS (`gromacs_density`), Amber
  (`amber_density`, bring-your-own asset — see §4).
- **Spawners:** density, voronoi, lof, fps, we, msm.
- **CV spaces:** fixed, pca, tica (+ tvae when torch is installed).
- **Subsystems:** in-loop MSM (`openmm_tica_msm`), MSM-guided spawning
  (`openmm_msm_spawn`), VAMP-2 feature selection, adaptive binning, **resume**
  (baseline), and **path reconstruction** (baseline).

**SKIP is not failure.** A feature whose optional backend is missing is skipped
with the reason (e.g. `missing: torch` for `tvae`, `missing: pmemd` for Amber,
`missing: shapely` for voronoi, `missing: gmx` for GROMACS). Install the backend
(`env.yml` provides all of them) to turn a SKIP into a real run.

### Reading the results

```bash
cat results/local/summary.json                       # overall table
cat results/local/local_<feature>/validate.json      # per-feature checks
```

`validate.json → overall` is `pass`/`fail`; on `fail`, the `checks` list names the
failing `code`s. Then:

1. Open `results/local/<feature>/run.log` (the CLI stdout/stderr) and read the
   tail for the actual error.
2. Look the failing `code` up in [`DEBUGGING.md`](DEBUGGING.md).
3. Fix, then re-run just that feature: `python hpc_tests/run_local_matrix.py
   --only <feature>`.

### Feature-check codes (from `checks/validate_results.py`)

Beyond the core codes (`TRAJ_FILES`, `CHECKPOINTS`, …), the feature runs assert:

| Code | Feature | Meaning of a failure |
| --- | --- | --- |
| `LATENT_DIM` | learned CV | `cvs.npz` latent dimension ≠ configured `adaptive_model.latent_dim`. |
| `MSM_NPZ` | in-loop MSM | no `iter_*/msm.npz`, or non-finite timescales / non-row-stochastic `T`. |
| `RESUME_CHAIN` | resume | the delta-checkpoint chain does not reconstruct to a gapless history. |
| `PATH_OUTPUT` | path | `trails-md-path` produced no / an empty output trajectory. |

---

## 2. Reproduce / confirm the correctness fixes

The three July-2026 review fixes have dedicated regression tests:

```bash
python -m pytest tests/test_review_fixes.py -q      # A1 MSM segmentation, A2 index sync, A3 RNG
python -m pytest tests/test_engine_isolation.py -q  # B1/B2 thread & device isolation
```

The A1 bug (GROMACS writes `step//stride + 1` frames while OpenMM/Amber write
`step//stride`) is visible directly: run `gromacs_density` and `openmm_fixed_density`
from the matrix and compare `frames_this_iteration / successful_walkers` in each
`run/output.log` — GROMACS yields one extra frame per walker, which is exactly why
MSM segmentation must use the stored per-walker frame records, not a constant.

---

## 3. Scheduler (SLURM / PBS) matrix — on a cluster

Two options, from the fastest smoke test to the full feature matrix:

```bash
# (a) 4-way scheduler smoke test (OpenMM alanine dipeptide, CPU + GPU):
sbatch hpc_tests/slurm/run_cpu.sbatch
sbatch hpc_tests/slurm/run_gpu.sbatch      # PBS: qsub hpc_tests/pbs/run_{cpu,gpu}.pbs

# (b) full FEATURE matrix under the scheduler (same features as §1):
#     edit the SITE SETUP block + TRAILS_HPC_* vars in the driver first.
sbatch hpc_tests/slurm/run_features.sbatch # PBS: qsub hpc_tests/pbs/run_features.pbs
```

The feature driver runs `run_local_matrix.py --backend {slurm,pbs}`, which injects
an `execution` block into every feature config from these environment variables:

| Var | Meaning | Default |
| --- | --- | --- |
| `TRAILS_HPC_PARTITION` | SLURM partition / PBS queue | (site default) |
| `TRAILS_HPC_ACCOUNT` | allocation/account | — |
| `TRAILS_HPC_WALLTIME` | per-walker walltime | `00:20:00` |
| `TRAILS_HPC_CPUS` | `cpus_per_task` (also caps OpenMM/GROMACS threads) | `2` |
| `TRAILS_HPC_GPUS` | `gpus_per_task` (set `1` to exercise GPU binding) | `0` |
| `TRAILS_HPC_MODULES` | comma-separated `module load …` lines for the array jobs | — |

The per-walker array jobs run in **fresh shells**, so `TRAILS_HPC_MODULES` (or the
config's `execution.module_loads`) must fully reconstruct the runtime environment.
Reports land under `results/{slurm,pbs}_features/`, same structure as §1, and are
triaged the same way via [`DEBUGGING.md`](DEBUGGING.md).

**GPU device isolation** is checked automatically on the GPU path. With
`TRAILS_HPC_GPUS=1` the OpenMM features run on CUDA and the validator adds the
`GPU_BINDING` check (from the per-walker `<traj>.gpu.json` markers): it fails on a
silent CUDA→CPU fallback, and — when you also set `TRAILS_HPC_GPU_COUNT` to the
number of GPUs — fails if walkers do not spread across `min(walkers, GPUs)`
devices. The `run_gpu.{sbatch,pbs}` smoke scripts run the same check (set
`GPU_COUNT` to enforce spread). See [`DEBUGGING.md`](DEBUGGING.md) `GPU_BINDING`.

**Torque:** the array backend targets SLURM and PBS Pro / OpenPBS only.
`preflight.py --scheduler pbs` warns if it detects classic Torque; there, run the
`local` backend inside one multi-GPU allocation instead (see `DEBUGGING.md`
`PBS_FLAVOR`).

Preflight before a big run:

```bash
python hpc_tests/checks/preflight.py --scheduler slurm --config hpc_tests/configs/alad_cpu_slurm.yaml
```

---

## 4. Amber (bring-your-own asset)

The repo ships no self-contained Amber `prmtop`/`rst7` (building one needs
`tleap`/ParmEd). The `amber_density` feature therefore **SKIPs** (not FAILs)
until both `pmemd`/`pmemd.cuda`/`sander` is on PATH **and** the asset exists at
`examples/alanine_dipeptide/alad.{prmtop,rst7}` (the `amber_asset` capability).
To exercise the Amber engine:

1. Generate the asset from the committed OpenMM system (Amber14 vacuum), so the
   OpenMM and Amber engines run the identical physics:

   ```bash
   python hpc_tests/assets/build_alad_amber.py     # -> examples/alanine_dipeptide/alad.{prmtop,rst7}
   ```

   This uses **ParmEd**. If ParmEd is unavailable it prints (and writes) a
   `tleap` recipe (`leaprc.protein.ff14SB`, `saveamberparm`) as a fallback.
2. Ensure `pmemd` (or `pmemd.cuda`/`sander`) is on PATH.
3. `python hpc_tests/run_local_matrix.py --only amber_density`.

Amber trajectory format is auto-detected (`.nc` for `pmemd.cuda`, else `.mdcrd`);
pass `--amber-format ascii` to `validate_results.py` if you force ASCII output.

---

## 5. Reporting back

When you have a diagnosis, produce:

1. The failing `code`(s) and the single root cause.
2. Whether it is a **site/environment** issue (fix the YAML / modules / install a
   backend) or a **code** bug (propose a patch + a regression test — mirror
   `tests/test_review_fixes.py` for core logic or `tests/test_execution.py` for
   scheduler logic, both of which run off-cluster).
3. The relevant excerpt from `run.log` and, for scheduler runs, the per-walker
   `result_*.json` `traceback` under `<outdir>/iter_*/_jobs/`.
4. The exact `file:line` in `trails_md/` implicated.
