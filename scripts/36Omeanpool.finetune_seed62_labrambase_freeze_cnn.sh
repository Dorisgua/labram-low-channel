#!/usr/bin/env bash
set -euo pipefail
# SEED cross-subject full-channel baseline using LaBraM mean pooling instead
# of the AdaBrain all-token classification head.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/36Oada.finetune_seed62_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="seed62"
export CLASSIFIER_MODE="mean_pool"
export EXP_GROUP="${EXP_GROUP:-preexp36_seed62_mean_pool_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29512}"

exec bash "${BASE_SCRIPT}"
