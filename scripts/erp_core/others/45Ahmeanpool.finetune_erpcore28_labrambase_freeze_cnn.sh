#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# ERP CORE static prototype baseline:
# use 12 observed channels, fill the missing channels with fixed erpcore28 prototypes,
# then classify on the completed 28-channel token sequence.
export CHANNEL_SUBSET="${CHANNEL_SUBSET:-erpcore12}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-erpcore12_with_erpcore28}"
export POOLING_SCOPE="${POOLING_SCOPE:-high}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"

export FREEZE_CNN="${FREEZE_CNN:-1}"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-mean_pool}"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export EXP_GROUP="${EXP_GROUP:-${SCRIPT_NAME%.sh}}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"

exec bash "${SCRIPT_DIR}/45Omeanpool.finetune_erpcore28_labrambase_freeze_cnn.sh" "$@"
