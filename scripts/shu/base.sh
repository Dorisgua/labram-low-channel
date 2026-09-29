#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export DATASET="SHU"
export PRELOAD_DATA="${PRELOAD_DATA:-1}"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/AdaBrain-Bench-main_film/preprocessing/SHU/cross_subject_json}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export NORM_METHOD="${NORM_METHOD:-95}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"

[[ -f "${DATA_PATH}/train.json" ]] || { echo "Missing SHU train manifest: ${DATA_PATH}/train.json" >&2; exit 1; }
exec bash "${REPO_DIR}/scripts/base.sh" "$@"
