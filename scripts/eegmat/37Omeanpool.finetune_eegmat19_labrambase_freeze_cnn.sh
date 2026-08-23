#!/usr/bin/env bash
set -euo pipefail

# EEGMAT O mean-pool freeze-CNN wrapper: preserve the original bias settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Oada.finetune_eegmat19_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="eegmat19"
export COMPLETION_SCOPE="none"
export CHANNEL_PROTOTYPE_PATH=""
export POOLING_SCOPE="low"
export FREEZE_CNN="1"
export CLASSIFIER_MODE="mean_pool"
export CLASSIFIER_TOKEN_SCOPE="all"
export DISABLE_REL_POS_BIAS="1"
export DISABLE_QKV_BIAS="1"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat19_mean_pool_cross_subject}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export MASTER_PORT="${MASTER_PORT:-29517}"

exec bash "${BASE_SCRIPT}" "$@"
