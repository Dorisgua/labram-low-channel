#!/usr/bin/env bash
set -euo pipefail

# TUEV O freeze-CNN wrapper: use all 23 real channels without completion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh"

export CHANNEL_SUBSET="tuev23"
export COMPLETION_SCOPE="none"
export POOLING_SCOPE="low"
export CHANNEL_PROTOTYPE_PATH=""
export FREEZE_CNN="1"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"

exec bash "${BASE_SCRIPT}" "$@"
