#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Full finetuning on the CBraMod cross-subject split.
export DATASET="bciiv2a_cbramod"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/data_splits/bciiv2a_cbramod_cross_subject_json}"
export EPOCHS="${EPOCHS:-50}"
export SEED="${SEED:-0}"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-bciiv2a_cbramod_O_full_finetune}"
export CHANNEL_SUBSET="bciiv2a22" COMPLETION_SCOPE="none" POOLING_SCOPE="low" FREEZE_CNN="0"
export CLASSIFIER_MODE="adabrain_all_token" CLASSIFIER_TOKEN_SCOPE="real"
exec bash "${SCRIPT_DIR}/../base.sh" "$@"
