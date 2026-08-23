#!/usr/bin/env bash
set -euo pipefail
# SEED-V: complete 23 real channels to 62 but mean-pool only the 23 real channels.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHANNEL_SUBSET="seedv23"
export COMPLETION_SCOPE="seedv23_with_seedv62"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth}"
export POOLING_SCOPE="low"
export EXP_GROUP="preexp35_seedv23_with_seedv62_prototype_low_pool_freeze_cnn"
export MASTER_PORT="${MASTER_PORT:-29519}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
exec bash "${SCRIPT_DIR}/35O.finetune_seedv62_labrambase_freeze_cnn.sh"
