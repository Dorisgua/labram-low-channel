#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# EEGMAT O mean-pool full-finetune wrapper: preserve the original bias settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Oada.finetune_eegmat19_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="eegmat19"
export COMPLETION_SCOPE="none"
export CHANNEL_PROTOTYPE_PATH=""
export POOLING_SCOPE="low"
export FREEZE_CNN="0"
export CLASSIFIER_MODE="mean_pool"
export CLASSIFIER_TOKEN_SCOPE="all"
export DISABLE_REL_POS_BIAS="1"
export DISABLE_QKV_BIAS="1"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat19_mean_pool_full_finetune_cross_subject}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export MASTER_PORT="${MASTER_PORT:-29521}"

exec bash "${BASE_SCRIPT}" "$@"
