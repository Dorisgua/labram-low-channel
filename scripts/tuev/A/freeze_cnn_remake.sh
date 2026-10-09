#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-tuev_A_freeze_cnn_remake}"
export CHANNEL_SUBSET="tuev13" COMPLETION_SCOPE="tuev13_with_tuev23" POOLING_SCOPE="high" FREEZE_CNN="1"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth}"
export CLASSIFIER_MODE="mean_pool" CLASSIFIER_TOKEN_SCOPE="all"
exec bash "${SCRIPT_DIR}/../base_remake.sh" "$@"
