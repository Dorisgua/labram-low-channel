#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# A baseline on the same CBraMod split and prototypes as Dynamic Stage 2.
export DATASET="bciiv2a_cbramod"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/data_splits/bciiv2a_cbramod_cross_subject_json}"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-bciiv2a_cbramod_A_freeze_cnn}"
export CHANNEL_SUBSET="bciiv2a13" COMPLETION_SCOPE="bciiv2a13_with_bciiv2a22" POOLING_SCOPE="high" FREEZE_CNN="1"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_bciiv2a22_cbramod_train_cnn_patch_embed_mean.pth}"
export EPOCHS="${EPOCHS:-50}"
export SEED="${SEED:-0}"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
