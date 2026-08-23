#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp40 Siena prototype completion wrapper:
# use 13 real Siena channels, complete to Siena-29 with train-set CNN
# patch_embed prototypes, use AdaBrain all-token head, freeze CNN.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/40Oada.finetune_siena29_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-siena13}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-siena13_with_siena29}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_siena29_cnn_patch_embed_mean.pth}"
export FREEZE_CNN="1"
export EXP_GROUP="${EXP_GROUP:-preexp40_siena13_with_siena29_adabrain_freeze_cnn_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29547}"

exec bash "${BASE_SCRIPT}" "$@"
