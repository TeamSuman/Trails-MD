#!/usr/bin/env bash
#SBATCH --job-name=trails-hpc-cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=trails_hpc_cpu_%j.out
#
# Driver job: SLURM, CPU-only queue.
#
# This lightweight driver runs the Trails-MD orchestrator, which in turn submits
# one *array job per iteration* to run the walkers. It performs preflight ->
# --check -> run -> validate and writes structured JSON to results/.
#
# EDIT the SITE SETUP block below, then submit from the repo root:
#     sbatch hpc_tests/slurm/run_cpu.sbatch
set -uo pipefail

# =========================== SITE SETUP (EDIT) ===============================
# module purge
# module load anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate trails-md
# NOTE: these SAME module loads / activation must be mirrored in the config's
#       execution.module_loads so the per-walker array jobs can import trails_md.
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-hpc_tests/configs/alad_cpu_slurm.yaml}"
ITERATIONS="${ITERATIONS:-3}"
WALKERS="${WALKERS:-8}"
RESULTS="${RESULTS:-results/slurm_cpu}"
mkdir -p "$RESULTS"

echo "== [1/4] preflight =="
python hpc_tests/checks/preflight.py --scheduler slurm --config "$CONFIG" \
    --out "$RESULTS/preflight.json" || { echo "PREFLIGHT FAILED"; exit 1; }

echo "== [2/4] trails-md --check =="
trails-md --config "$CONFIG" --check || { echo "CHECK FAILED"; exit 1; }

echo "== [3/4] run ($ITERATIONS iterations) =="
trails-md --config "$CONFIG" --iterations "$ITERATIONS" --log-level INFO \
    2>&1 | tee "$RESULTS/run.log"
RUN_RC=${PIPESTATUS[0]}
echo "run exit code: $RUN_RC"

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
exit $(( RUN_RC != 0 || VAL_RC != 0 ))
