#!/usr/bin/env bash
set -euo pipefail

# PreExp37 EEGMAT prototype completion:
# use 8 real EEGMAT channels, complete missing channels to EEGMAT-19 with
# train-set CNN patch_embed prototypes, then use mean-pooling classifier.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="eegmat8"
export COMPLETION_SCOPE="eegmat8_with_eegmat19"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_eegmat19_cnn_patch_embed_mean.pth}"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat8_with_eegmat19_mean_pool_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29519}"

exec bash "${BASE_SCRIPT}" "$@"
