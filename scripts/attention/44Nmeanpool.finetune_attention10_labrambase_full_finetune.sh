#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp44 Attention 10-channel full fine-tuning wrapper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/44Omeanpool.finetune_attention26_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-attention10}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
export EXP_GROUP="${EXP_GROUP:-preexp44_attention10_mean_pool_full_finetune_brainpro_split}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29548}"

exec bash "${BASE_SCRIPT}" "$@"
