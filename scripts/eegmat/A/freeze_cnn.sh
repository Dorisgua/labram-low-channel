#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-eegmat_A_freeze_cnn}"
export CHANNEL_SUBSET="eegmat8" COMPLETION_SCOPE="eegmat8_with_eegmat19" POOLING_SCOPE="high" FREEZE_CNN="1"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_eegmat19_cnn_patch_embed_mean.pth}"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
