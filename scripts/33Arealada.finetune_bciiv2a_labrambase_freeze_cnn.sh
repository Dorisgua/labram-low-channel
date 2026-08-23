#!/usr/bin/env bash
set -euo pipefail

# Keep 22 completed channels inside the Transformer, but let the AdaBrain
# flattened task head read only CLS + the 13 real input-channel tokens.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLASSIFIER_TOKEN_SCOPE="real"
export RUN_PREFIX_OVERRIDE="33Arealada.finetune_bciiv2a_labrambase_freeze_cnn"

exec bash "${SCRIPT_DIR}/33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh" "$@"
