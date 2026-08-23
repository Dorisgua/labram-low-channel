#!/usr/bin/env bash
set -euo pipefail

# PreExp40 Siena 13-channel AdaBrain freeze-CNN wrapper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/40Oada.finetune_siena29_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-siena13}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
export FREEZE_CNN="1"
export EXP_GROUP="${EXP_GROUP:-preexp40_siena13_adabrain_freeze_cnn_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29546}"

exec bash "${BASE_SCRIPT}" "$@"
