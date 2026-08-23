#!/usr/bin/env bash
set -euo pipefail
# SEED cross-subject reduced-channel baseline: retain the fixed 23-channel
# symmetric montage, remove 39 channels, and perform no channel completion.
# Reuse the audited 36O launcher so optimization and logging stay identical.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/36Oada.finetune_seed62_labrambase_freeze_cnn.sh"
export CHANNEL_SUBSET="seed23"
export COMPLETION_SCOPE="none"
export CHANNEL_PROTOTYPE_PATH=""
export POOLING_SCOPE="low"
export FREEZE_CNN="1"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"
export EXP_GROUP="${EXP_GROUP:-preexp36_seed23_cross_subject}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export MASTER_PORT="${MASTER_PORT:-29511}"

exec bash "${BASE_SCRIPT}" "$@"
