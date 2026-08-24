#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# BCI-IV-2a N full-finetune wrapper: use 13 real channels without completion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/33Arealada.finetune_bciiv2a_labrambase_freeze_cnn.sh"

export COMPLETION_SCOPE="none"
export POOLING_SCOPE="low"
export FREEZE_CNN="0"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"

exec bash "${BASE_SCRIPT}" "$@"
