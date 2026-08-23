#!/usr/bin/env bash
set -euo pipefail

# PhysioNet O full-finetune wrapper: reuse the 64-channel command without freezing CNN.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/34Oeegfm.finetune_physionet_labrambase_freeze_cnn.sh"

export EPOCHS="${EPOCHS:-50}"
export FREEZE_CNN="0"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"

exec bash "${BASE_SCRIPT}" "$@"
