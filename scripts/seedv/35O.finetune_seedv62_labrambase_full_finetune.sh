#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# SEED-V 62-channel full fine-tuning: train CNN, Transformer, and head.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHANNEL_SUBSET="seedv62"
export COMPLETION_SCOPE="none"
export POOLING_SCOPE="low"
export FREEZE_CNN="0"
export EXP_GROUP="preexp35_seedv62_full_finetune"
export MASTER_PORT="${MASTER_PORT:-29522}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
exec bash "${SCRIPT_DIR}/35O.finetune_seedv62_labrambase_freeze_cnn.sh"
