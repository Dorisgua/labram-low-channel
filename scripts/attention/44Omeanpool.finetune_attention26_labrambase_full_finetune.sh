#!/usr/bin/env bash
set -euo pipefail

# PreExp44 Attention full fine-tuning wrapper:
# reuse the Attention26 mean-pool launcher, but unfreeze CNN/patch_embed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/44Omeanpool.finetune_attention26_labrambase_freeze_cnn.sh"

export FREEZE_CNN="0"
export EXP_GROUP="${EXP_GROUP:-preexp44_attention26_mean_pool_full_finetune_brainpro_split}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29545}"

exec bash "${BASE_SCRIPT}" "$@"
