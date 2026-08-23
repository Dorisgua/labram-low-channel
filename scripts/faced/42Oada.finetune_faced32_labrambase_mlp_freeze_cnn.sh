#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# FACED close-to-CBraMod LaBraM variant:
# 32 channels, CBraMod split and data/100 normalization, flattened token MLP
# head, frozen CNN/patch_embed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/42Omeanpool.finetune_faced32_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-faced32}"
export NORM_METHOD="${NORM_METHOD:-0.1mv}"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_mlp_token}"
export FREEZE_CNN="1"
export LR="${LR:-1e-4}"
export DROP="${DROP:-0.1}"
export EXP_GROUP="${EXP_GROUP:-preexp42_faced32_adabrain_mlp_cbramod_split_freeze_cnn}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29525}"

exec bash "${BASE_SCRIPT}" "$@"
