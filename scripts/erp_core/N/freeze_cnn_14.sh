#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-erp_core_N_freeze_cnn}"
export CHANNEL_SUBSET="erpcore14" COMPLETION_SCOPE="none" POOLING_SCOPE="low" FREEZE_CNN="1"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-erpcore14_with_erpcore28}"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
