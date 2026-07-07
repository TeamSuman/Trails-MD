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
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-hpc_tests/configs/alad_gpu_slurm.yaml}"
ITERATIONS="${ITERATIONS:-3}"
WALKERS="${WALKERS:-8}"
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
    --engine openmm --out "$RESULTS/validate.json"
VAL_RC=$?
echo "SUMMARY: run_rc=$RUN_RC validate_rc=$VAL_RC results in $RESULTS/"
# GPU-binding evidence: grep the walker logs to confirm distinct devices were used.
echo "Tip: inspect $OUTDIR/iter_*/_jobs/logs_attempt*/ for per-walker CUDA device lines."
exit $(( RUN_RC != 0 || VAL_RC != 0 ))
