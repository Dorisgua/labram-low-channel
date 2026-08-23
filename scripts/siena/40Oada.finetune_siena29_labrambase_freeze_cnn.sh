#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp40 Siena AdaBrain-style freeze-CNN wrapper:
# reuse the Siena29 all-token launcher, but freeze CNN/patch_embed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/40Oada.finetune_siena29_labrambase_full_finetune.sh"

export FREEZE_CNN="1"
export EXP_GROUP="${EXP_GROUP:-preexp40_siena29_adabrain_freeze_cnn_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29542}"

exec bash "${BASE_SCRIPT}" "$@"
