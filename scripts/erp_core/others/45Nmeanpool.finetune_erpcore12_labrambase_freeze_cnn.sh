#!/usr/bin/env bash
set -euo pipefail
# Wrapper for the ERP CORE 12-channel LaBraM mean-pooling baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
BASE_SCRIPT="${SCRIPT_DIR}/45Omeanpool.finetune_erpcore28_labrambase_freeze_cnn.sh"

export FREEZE_CNN="1"
export CHANNEL_SUBSET="${CHANNEL_SUBSET:-erpcore12}"
export EXP_GROUP="${EXP_GROUP:-${SCRIPT_NAME%.sh}}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
export MASTER_PORT="${MASTER_PORT:-29547}"

exec bash "${BASE_SCRIPT}" "$@"

