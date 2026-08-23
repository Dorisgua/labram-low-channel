#!/usr/bin/env bash
set -euo pipefail
# SEED 36Ah: 23 real channels completed to 62 with train-set CNN prototypes;
# mean-pool all 62 real/completed channel tokens (high pooling).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/36Oada.finetune_seed62_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="seed23"
export COMPLETION_SCOPE="seed23_with_seed62"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_seed62_cnn_patch_embed_mean.pth}"
export POOLING_SCOPE="high"
export CLASSIFIER_MODE="mean_pool"
export EXP_GROUP="${EXP_GROUP:-preexp36_seed23_with_seed62_mean_pool_high_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29514}"

exec bash "${BASE_SCRIPT}"
