#!/usr/bin/env bash
set -euo pipefail

# Stage1 preset: full-channel run with two contrastive losses.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX="${RUN_PREFIX:-full_two_contrastive}"
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export OPT_EPS="${OPT_EPS:-1e-8}"
export OPTIMIZER="${OPTIMIZER:-adamw}"
export SCHEDULE="${SCHEDULE:-cosine}"
export TEMPERATURE="${TEMPERATURE:-0.2}"

export SCOPE="${SCOPE:-full}"
export MISSING_FILL="${MISSING_FILL:-not_applicable}"
export OUTPUT_BASE="${OUTPUT_BASE:-none}"
export COMPONENT_MODE="${COMPONENT_MODE:-identity}"
export COMPOSITION_MODE="${COMPOSITION_MODE:-sum}"
export SUB_CONTRA_WEIGHT="${SUB_CONTRA_WEIGHT:-1}"
export TASK_CONTRA_WEIGHT="${TASK_CONTRA_WEIGHT:-1}"
export RECON_WEIGHT="${RECON_WEIGHT:-0}"
export SAMPLING="${SAMPLING:-cslpae}"

exec "${SCRIPT_DIR}/train_base.sh" "$@"
