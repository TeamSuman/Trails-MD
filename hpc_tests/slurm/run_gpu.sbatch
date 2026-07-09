#!/usr/bin/env bash
#SBATCH --job-name=trails-hpc-gpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=trails_hpc_gpu_%j.out
#
# Driver job: SLURM, GPU queue. The driver itself needs no GPU — it orchestrates
# and submits one array job per iteration whose elements each request a GPU
# (gpus_per_task in the config). This validates per-walker GPU binding.
#
# EDIT the SITE SETUP block, then:  sbatch hpc_tests/slurm/run_gpu.sbatch
set -uo pipefail

# =========================== SITE SETUP (EDIT) ===============================
# module purge
# module load cuda/12.2
# module load anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate trails-md
# Mirror these in the config's execution.module_loads.
#
# Choosing the MD engine: OpenMM is used here (engine.md_engine: openmm). To use
# GROMACS or Amber instead, `module load` the engine above (and in the config's
# execution.module_loads for the array jobs), then set the executable in the
# config: engine.gromacs_executable / engine.amber_executable (a name on PATH or
# an absolute path), and for GROMACS engine.gromacs_include_dir.
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-hpc_tests/configs/alad_gpu_slurm.yaml}"
ITERATIONS="${ITERATIONS:-3}"
WALKERS="${WALKERS:-8}"
GPU_COUNT="${GPU_COUNT:-}"          # distinct GPUs available; set to enforce walker spread
RESULTS="${RESULTS:-results/slurm_gpu}"
mkdir -p "$RESULTS"

echo "== [1/4] preflight =="
python hpc_tests/checks/preflight.py --scheduler slurm --gpu --config "$CONFIG" \
    --out "$RESULTS/preflight.json" || { echo "PREFLIGHT FAILED"; exit 1; }

echo "== [2/4] trails-md --check =="
trails-md --config "$CONFIG" --check || { echo "CHECK FAILED"; exit 1; }

echo "== [3/4] run ($ITERATIONS iterations) =="
trails-md --config "$CONFIG" --iterations "$ITERATIONS" --log-level INFO \
    2>&1 | tee "$RESULTS/run.log"
RUN_RC=${PIPESTATUS[0]}

echo "== [4/4] validate =="
OUTDIR=$(python - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
out = cfg.get("outdir", "runs/sampler_output")
base = Path(sys.argv[1]).parent
print(out if Path(out).is_absolute() else str((base / out).resolve()))
PY
)
python hpc_tests/checks/validate_results.py --outdir "$OUTDIR" \
    --expect-iterations "$ITERATIONS" --expect-walkers "$WALKERS" \
    --engine openmm --check-gpu-binding ${GPU_COUNT:+--gpu-count "$GPU_COUNT"} \
    --out "$RESULTS/validate.json"
VAL_RC=$?
echo "SUMMARY: run_rc=$RUN_RC validate_rc=$VAL_RC results in $RESULTS/"
# GPU_BINDING is now validated automatically from the per-walker <traj>.gpu.json
# markers (catches a silent CUDA->CPU fallback; with GPU_COUNT set, also catches
# all walkers piling onto one device). For manual inspection, the raw device each
# walker used is in $OUTDIR/iter_*/*.gpu.json and the array logs under
# $OUTDIR/iter_*/_jobs/logs_attempt*/.
exit $(( RUN_RC != 0 || VAL_RC != 0 ))
