#!/usr/bin/env bash
#SBATCH --job-name=autosampler-driver
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=autosampler_driver_%j.out
#
# Driver job for AutoSampler on a SLURM cluster.
#
# AutoSampler itself submits one *array job per iteration* (sbatch) to run the
# walkers, so this driver is lightweight: it just runs the orchestrator. Set
# `execution.backend: slurm` in your config (see examples/AIB9/
# config_msm_feature_selection.yaml and docs/execution.md).
#
# Submit with:  sbatch examples/slurm_submit.sh
set -euo pipefail

# --- Make the autosampler package importable in walker jobs too -------------
# These same module loads / env activation should be mirrored in the config's
# execution.module_loads so the per-walker array jobs can import autosampler.
module purge
# module load cuda/12.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autosampler

CONFIG="${CONFIG:-examples/AIB9/config_msm_vampnet.yaml}"
ITERATIONS="${ITERATIONS:-200}"

autosampler --config "${CONFIG}" --check
autosampler --config "${CONFIG}" --iterations "${ITERATIONS}" --log-level INFO
