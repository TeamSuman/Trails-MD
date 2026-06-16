#!/usr/bin/env bash
#PBS -N autosampler-driver
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=24:00:00
#PBS -j oe
#
# Driver job for AutoSampler on a PBS / Torque (PBS Pro) cluster.
#
# AutoSampler submits one *array job per iteration* (qsub) for the walkers, so
# this driver only runs the orchestrator. Set `execution.backend: pbs` in your
# config (see docs/execution.md).
#
# Submit with:  qsub examples/pbs_submit.sh
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

# --- Make the autosampler package importable in walker jobs too -------------
# Mirror these in the config's execution.module_loads.
# module load openmm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autosampler

CONFIG="${CONFIG:-examples/AIB9/config_msm_vampnet.yaml}"
ITERATIONS="${ITERATIONS:-200}"

autosampler --config "${CONFIG}" --check
autosampler --config "${CONFIG}" --iterations "${ITERATIONS}" --log-level INFO
