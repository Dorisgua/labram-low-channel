#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME

# PreExp39 HGD reduced-channel baseline: use the 20-channel motor-cortex subset
# while reusing the audited HGD mean-pooling launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/39Omeanpool.finetune_hgd78_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="hgd20"
export EXP_GROUP="${EXP_GROUP:-preexp39_hgd20_mean_pool_official_split}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29520}"

exec bash "${BASE_SCRIPT}" "$@"
