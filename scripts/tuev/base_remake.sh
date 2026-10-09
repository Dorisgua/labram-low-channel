#!/usr/bin/env bash
set -euo pipefail

# Classification defaults matching the historical 17Ah entrypoint.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export UPDATE_FREQ="${UPDATE_FREQ:-8}"
export LR="${LR:-5e-4}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export LAYER_DECAY="${LAYER_DECAY:-0.65}"
export DROP_PATH="${DROP_PATH:-0.1}"
export SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
export SEED="${SEED:-0}"
export SMOOTHING="${SMOOTHING:-0.1}"
export BEST_METRIC="${BEST_METRIC:-cohen_kappa}"
export CLASSIFIER_MODE="mean_pool" CLASSIFIER_TOKEN_SCOPE="all"
export DISABLE_REL_POS_BIAS="${DISABLE_REL_POS_BIAS:-1}"
export DISABLE_QKV_BIAS="${DISABLE_QKV_BIAS:-1}"
export PRELOAD_DATA=1
exec bash "${SCRIPT_DIR}/base.sh" "$@"
