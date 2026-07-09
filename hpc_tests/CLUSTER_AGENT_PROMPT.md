# Prompt for the SLURM-cluster Claude Code agent

Copy the block below into a Claude Code session **running on the SLURM test
cluster** (a login/submit node with a shared filesystem). It is self-contained;
fill the three angle-bracket placeholders first.

---

You are running on a SLURM HPC cluster to **test, benchmark, and harden Trails-MD**
— an adaptive molecular-dynamics sampling framework — across all major features
and all three MD engines (OpenMM/CUDA, GROMACS, Amber), on both GPU and CPU nodes.
You are authorized to **modify the code, configs, and test scripts** as needed to
make the tests pass and to improve HPC performance, then report what you changed
and why. Work on a shared scratch/project path, never node-local `/tmp`.

Site specifics (fill these in):
- SLURM partitions: CPU=`<cpu-partition>`, GPU=`<gpu-partition>`; account=`<account>`.
- Module loads to make the env importable in a fresh shell (e.g.
  `module load cuda/12.x anaconda3`; plus `gromacs/...` and `amber/...` when
  testing those engines).

## 0. Setup

```bash
git clone <repo-url> && cd Trails-MD
git checkout hpc-test-readiness      # the HPC-readiness branch / PR
# Editable install with test extras (or: conda env create -f env.yml && conda activate trails-md)
pip install -e ".[openmm,test]"
python -c "import trails_md; print('ok')"
```

Read `hpc_tests/RUNBOOK.md` (ordered procedure), `hpc_tests/DEBUGGING.md` (failure
codes → fixes), `docs/execution.md` (engine/backend selection), and
`docs/performance.md` (GPU-utilization plan) before starting.

## 1. Off-cluster sanity (fast, no scheduler)

```bash
python -m pytest -q
python hpc_tests/run_local_matrix.py --list          # capability + feature matrix
python hpc_tests/run_local_matrix.py                 # local backend, all runnable features
```
Everything runnable should PASS; missing backends SKIP (that is fine). To exercise
Amber, build its asset first: `python hpc_tests/assets/build_alad_amber.py`, then
`module load amber` and rerun. To turn a `deep-tica`/`vampnet`/`spib` SKIP into a
run, install its backend (`mlcolvar`/`lightning`, `torch`, `deeptime`).

## 2. SLURM smoke tests (CPU then GPU)

Edit the SITE SETUP block (module loads + `conda activate`) in the driver scripts
and set `execution.partition/account/module_loads` in the `hpc_tests/configs/alad_*`
YAMLs, then:

```bash
python hpc_tests/checks/preflight.py --scheduler slurm --config hpc_tests/configs/alad_cpu_slurm.yaml
sbatch hpc_tests/slurm/run_cpu.sbatch
GPU_COUNT=<gpus-per-node> sbatch hpc_tests/slurm/run_gpu.sbatch   # exercises + validates GPU binding
```
The GPU run validates per-walker device isolation automatically (`GPU_BINDING`
from the `<traj>.gpu.json` markers): it FAILS on a silent CUDA→CPU fallback and,
with `GPU_COUNT`, if walkers do not spread across devices.

## 3. Full feature matrix under SLURM (all features, GPU + CPU)

```bash
export TRAILS_HPC_PARTITION=<gpu-partition> TRAILS_HPC_ACCOUNT=<account>
export TRAILS_HPC_GPUS=1 TRAILS_HPC_GPU_COUNT=<gpus-per-node>
export TRAILS_HPC_MODULES="module load cuda/12.x,module load anaconda3"   # replayed in each array job
sbatch hpc_tests/slurm/run_features.sbatch          # OpenMM features on GPU; validators run per feature
# CPU pass: unset TRAILS_HPC_GPUS/GPU_COUNT and resubmit.
```
Reports land under `results/slurm_features/`, one directory per feature with
`validate.json` + `run.log`. Triage any FAIL by its `code` via `DEBUGGING.md`.

## 4. Priorities

1. **All features × engines pass under SLURM** on GPU and CPU nodes. Prove each
   engine works with a learned CV + in-loop MSM, not just density (`gromacs_tica_msm`,
   `amber_tica`, `openmm_*`).
2. **MSM convergence workflow** (`openmm_msm_convergence`): confirm the monitor runs,
   `convergence.json` is written, and implied timescales are finite/settling. Then
   **benchmark** it on a slightly larger workload (more walkers/steps/iterations) and
   report convergence behavior, lag-time sensitivity, and wall-time breakdown
   (`Runner` vs `Other` in `output.log`) — this feeds the manuscript revision.
3. **GPU utilization** (`docs/performance.md`): measure `Runner` time vs `step`
   (longer walkers), then try **CUDA MPS** to pack multiple walkers per GPU and
   report the throughput gain; note where per-walker startup dominates.

## 5. When something fails

Follow `DEBUGGING.md`: read `results/.../preflight.json` and `validate.json`, the
driver `run.log`, and per-walker `iter_*/_jobs/result_*.json` (`error`+`traceback`)
and `logs_attempt*/*.out`. Decide **site/config** (fix YAML/modules/driver) vs
**code bug** (patch + add a regression test — mirror `tests/test_execution.py` for
scheduler logic or `tests/test_review_fixes.py` for core logic, both off-cluster).
Keep test workloads tiny until green, then scale up.

## 6. Report back

Produce: (a) the PASS/SKIP/FAIL matrix for CPU and GPU; (b) for each FAIL, the
`code`, root cause, and the fix (site vs code, with the `file:line`); (c) any code
or script changes you made, with a one-line rationale each and the regression test
added; (d) the MSM-convergence and GPU-utilization benchmark numbers. Commit your
changes to a branch and open a PR (or push to `hpc-test-readiness`), summarizing the
above.

---
