#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp39 HGD prototype completion:
# use 20 real HGD motor-cortex channels, complete missing channels to HGD-78
# with train-set CNN patch_embed prototypes, then use mean-pooling classifier.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/39Omeanpool.finetune_hgd78_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="hgd20"
export COMPLETION_SCOPE="hgd20_with_hgd78"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_hgd78_cnn_patch_embed_mean.pth}"
export EXP_GROUP="${EXP_GROUP:-preexp39_hgd20_with_hgd78_mean_pool_official_split}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29521}"

exec bash "${BASE_SCRIPT}" "$@"
