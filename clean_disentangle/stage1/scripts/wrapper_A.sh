#!/usr/bin/env bash
set -euo pipefail
# Stage1 experiment A preset.
export EXP_NAME="${EXP_NAME:-full_d_only}"
export SCOPE="${SCOPE:-full}"
export MISSING_FILL="${MISSING_FILL:-not_applicable}"
export OUTPUT_BASE="${OUTPUT_BASE:-none}"
export COMPONENT_MODE="${COMPONENT_MODE:-identity}"
export COMPOSITION_MODE="${COMPOSITION_MODE:-sum}"
export SUB_CONTRA_WEIGHT="${SUB_CONTRA_WEIGHT:-1}"
export TASK_CONTRA_WEIGHT="${TASK_CONTRA_WEIGHT:-1}"
export SWAP_SUB_WEIGHT="${SWAP_SUB_WEIGHT:-1}"
export SWAP_TASK_WEIGHT="${SWAP_TASK_WEIGHT:-1}"
export RECON_WEIGHT="${RECON_WEIGHT:-0}"
export MISSING_MSE_WEIGHT="${MISSING_MSE_WEIGHT:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/base_stage1.sh" "$@"
