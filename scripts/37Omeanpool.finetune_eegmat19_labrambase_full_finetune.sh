#!/usr/bin/env bash
set -euo pipefail

# PreExp37 EEGMAT mean-pooling full-finetune wrapper:
# same as 37O except CNN/patch_embed is trainable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-eegmat19}"
export FREEZE_CNN="0"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat19_mean_pool_full_finetune_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29521}"

exec bash "${BASE_SCRIPT}" "$@"
