#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-physionet_A_freeze_cnn}"
export CHANNEL_SUBSET="physionet32" COMPLETION_SCOPE="physionet32_with_physionet64" POOLING_SCOPE="high" FREEZE_CNN="1"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_physionet64_cnn_patch_embed_mean.pth}"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
