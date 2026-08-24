#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-bciiv2a_N_freeze_cnn}"
export CHANNEL_SUBSET="bciiv2a13" COMPLETION_SCOPE="none" POOLING_SCOPE="low" FREEZE_CNN="1"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
