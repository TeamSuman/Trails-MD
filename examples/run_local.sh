#!/usr/bin/env bash
# Run Trails-MD locally (multi-GPU workstation or a single machine).
#
# Usage:
#   ./examples/run_local.sh [CONFIG] [ITERATIONS]
#
# Examples:
#   ./examples/run_local.sh examples/AlaD/config.yaml 20
#   ./examples/run_local.sh examples/AIB9/config_msm_vampnet.yaml 200
set -euo pipefail

CONFIG="${1:-examples/AIB9/config_msm_vampnet.yaml}"
ITERATIONS="${2:-200}"

# Optional: cap MD subprocess runtime (seconds) to catch hung GROMACS/Amber jobs.
export TRAILS_MD_TIMEOUT="${TRAILS_MD_TIMEOUT:-3600}"

# Validate inputs first (no MD is run).
trails-md --config "${CONFIG}" --check

# Run. With msm.enabled, this stops early once the MSM converges.
trails-md --config "${CONFIG}" --iterations "${ITERATIONS}" --log-level INFO

# To resume after an interruption:
#   trails-md --config "${CONFIG}" --resume --iterations "${ITERATIONS}"
