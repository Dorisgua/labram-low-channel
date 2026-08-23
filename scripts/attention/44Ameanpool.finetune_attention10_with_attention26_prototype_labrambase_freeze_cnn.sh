#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp44 Attention prototype completion wrapper:
# use 10 real Attention channels, complete to Attention-26 with train-set CNN
# patch_embed prototypes, use mean-pooling classifier, freeze CNN/patch_embed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/44Omeanpool.finetune_attention26_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-attention10}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-attention10_with_attention26}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_attention26_cnn_patch_embed_mean.pth}"
export FREEZE_CNN="1"
export EXP_GROUP="${EXP_GROUP:-preexp44_attention10_with_attention26_mean_pool_freeze_cnn_brainpro_split}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29549}"

exec bash "${BASE_SCRIPT}" "$@"
