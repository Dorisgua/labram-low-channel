#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# ERP Core full-electrode control: use all 28 real EEG channels without
# completion, and train the CNN, Transformer, and AdaBrain all-token head.
export SEED="${SEED:-0}"
export CHANNEL_SUBSET="${CHANNEL_SUBSET:-erpcore28}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
export POOLING_SCOPE="${POOLING_SCOPE:-high}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"

export FREEZE_CNN="0"
export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="all"
export ADABRAIN_PRESERVE_BACKBONE_NO_WEIGHT_DECAY="1"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export EXP_GROUP="${EXP_GROUP:-${SCRIPT_NAME%.sh}}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
export MASTER_PORT="${MASTER_PORT:-29549}"

exec bash "${SCRIPT_DIR}/45Omeanpool.finetune_erpcore28_labrambase_freeze_cnn.sh" "$@"
