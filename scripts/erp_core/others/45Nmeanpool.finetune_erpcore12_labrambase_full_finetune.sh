#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# Wrapper for the ERP CORE 12-channel LaBraM mean-pooling baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
BASE_SCRIPT="${SCRIPT_DIR}/45Omeanpool.finetune_erpcore28_labrambase_freeze_cnn.sh"

export SEED="${SEED:-2}"
export FREEZE_CNN="0"
export CHANNEL_SUBSET="${CHANNEL_SUBSET:-erpcore12}"
export EXP_GROUP="${EXP_GROUP:-${SCRIPT_NAME%.sh}}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
export MASTER_PORT="${MASTER_PORT:-29549}"

exec bash "${BASE_SCRIPT}" "$@"

