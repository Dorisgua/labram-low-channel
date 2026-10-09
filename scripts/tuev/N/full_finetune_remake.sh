#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-tuev_N_full_finetune_remake}"
export CHANNEL_SUBSET="tuev13" COMPLETION_SCOPE="none" POOLING_SCOPE="low" FREEZE_CNN="0"
export CLASSIFIER_MODE="mean_pool" CLASSIFIER_TOKEN_SCOPE="all"
exec bash "${SCRIPT_DIR}/../base_remake.sh" "$@"
