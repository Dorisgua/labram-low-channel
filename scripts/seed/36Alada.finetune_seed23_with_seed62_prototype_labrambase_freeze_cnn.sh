#!/usr/bin/env bash
set -euo pipefail
# SEED 36Alada: 23 real channels completed to 62 with train-set CNN
# prototypes, low pooling scope, AdaBrain flattened real-token head, frozen CNN.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/36Oada.finetune_seed62_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="seed23"
export COMPLETION_SCOPE="seed23_with_seed62"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_seed62_cnn_patch_embed_mean.pth}"
export POOLING_SCOPE="low"
export FREEZE_CNN="1"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
export EXP_GROUP="${EXP_GROUP:-preexp36_seed23_with_seed62_adabrain_low_cross_subject}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export MASTER_PORT="${MASTER_PORT:-29516}"

exec bash "${BASE_SCRIPT}" "$@"
