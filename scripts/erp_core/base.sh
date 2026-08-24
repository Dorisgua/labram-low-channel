#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export DATASET="ERPCORE"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/../CSLP-AE/data_preparation/simple_data.pt}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export DISABLE_REL_POS_BIAS="${DISABLE_REL_POS_BIAS:-1}"
export DISABLE_QKV_BIAS="${DISABLE_QKV_BIAS:-1}"

[[ -f "${DATA_PATH}" ]] || { echo "Missing ERP-Core data: ${DATA_PATH}" >&2; exit 1; }

exec bash "${REPO_DIR}/scripts/base.sh" "$@"
