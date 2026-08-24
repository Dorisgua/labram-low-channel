#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Head-only control for 45Ahada: use the same static 12 -> 28 prototype
# completion, but classify from all completed tokens with the Dynamic model's
# flattened, max-norm-constrained linear head.
export SEED="${SEED:-0}"
export CHANNEL_SUBSET="${CHANNEL_SUBSET:-erpcore12}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-erpcore12_with_erpcore28}"
export POOLING_SCOPE="${POOLING_SCOPE:-high}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"

export FREEZE_CNN="${FREEZE_CNN:-1}"
export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
# Preserve 45Ahada's optimizer exclusions so the intended change is the
# classification readout rather than the embedding weight-decay policy.
export ADABRAIN_PRESERVE_BACKBONE_NO_WEIGHT_DECAY="1"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export EXP_GROUP="${EXP_GROUP:-${SCRIPT_NAME%.sh}}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
export MASTER_PORT="${MASTER_PORT:-29548}"

exec bash "${SCRIPT_DIR}/45Oada.finetune_erpcore28_labrambase_freeze_cnn.sh" "$@"
