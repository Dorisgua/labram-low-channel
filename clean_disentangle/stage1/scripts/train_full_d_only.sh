#!/usr/bin/env bash
# Experiment A: Full + D only
set -euo pipefail

# Stage1 preset: full-channel reconstruction with D only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX="${RUN_PREFIX:-full_d_only}"
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-4}"
export UNFREEZE_CNN="${UNFREEZE_CNN:-0}"
export CNN_LR_MULT="${CNN_LR_MULT:-0.1}"
export MIN_LR="${MIN_LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export TEMPERATURE="${TEMPERATURE:-0.2}"

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
export SAMPLING="${SAMPLING:-cslpae}"

exec "${SCRIPT_DIR}/train_base.sh" "$@"
