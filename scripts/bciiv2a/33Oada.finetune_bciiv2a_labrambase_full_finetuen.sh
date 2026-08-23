#!/usr/bin/env bash
set -euo pipefail

# BCI-IV-2a O full-finetune wrapper: use all 22 real channels without completion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="bciiv2a22"
export COMPLETION_SCOPE="none"
export POOLING_SCOPE="low"
export FREEZE_CNN="0"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"

exec bash "${BASE_SCRIPT}" "$@"
