#!/usr/bin/env bash
#SBATCH --job-name=trails-features
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=trails_features_%j.out
#
# Driver job: run the full Trails-MD FEATURE MATRIX on SLURM. Each feature's
# walkers are dispatched as a SLURM array job (execution.backend=slurm). The
# driver itself only orchestrates. This is the cluster counterpart of
#   python hpc_tests/run_local_matrix.py            (local backend)
# and reuses the exact same feature definitions and validators.
#
# EDIT the SITE SETUP block and the TRAILS_HPC_* resource vars, then:
#   sbatch hpc_tests/slurm/run_features.sbatch
set -uo pipefail

# =========================== SITE SETUP (EDIT) ===============================
# module purge
# module load cuda/12.2
# module load anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate trails-md
#
# Engine modules for the array jobs go in TRAILS_HPC_MODULES below (fresh shells);
# GROMACS/Amber executables and paths are set per feature in run_local_matrix.py.
# =============================================================================

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# Per-site resources injected into every feature's execution block.
export TRAILS_HPC_PARTITION="${TRAILS_HPC_PARTITION:-}"   # e.g. cpu / gpu
export TRAILS_HPC_ACCOUNT="${TRAILS_HPC_ACCOUNT:-}"
export TRAILS_HPC_WALLTIME="${TRAILS_HPC_WALLTIME:-00:20:00}"
export TRAILS_HPC_CPUS="${TRAILS_HPC_CPUS:-2}"
export TRAILS_HPC_GPUS="${TRAILS_HPC_GPUS:-0}"           # set 1 to exercise GPU binding
export TRAILS_HPC_GPU_COUNT="${TRAILS_HPC_GPU_COUNT:-}"  # distinct GPUs available; enforces walker spread in GPU_BINDING
# module_loads reproduced inside each per-walker array job (fresh shells):
export TRAILS_HPC_MODULES="${TRAILS_HPC_MODULES:-}"      # e.g. "module load cuda/12.2,module load anaconda3"

RESULTS="${RESULTS:-results/slurm_features}"
ONLY_ARGS=""
for f in ${ONLY:-}; do ONLY_ARGS="$ONLY_ARGS --only $f"; done

python hpc_tests/run_local_matrix.py --backend slurm \
    --results-dir "$RESULTS" $ONLY_ARGS
RC=$?
echo "Feature matrix finished (rc=$RC). Reports under $RESULTS/ ; see hpc_tests/RUNBOOK.md."
exit $RC
