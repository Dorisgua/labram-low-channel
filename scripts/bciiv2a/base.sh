#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export DATASET="bciiv2a"
# export DATA_PATH="${DATA_PATH:-${REPO_DIR}/preprocessing/BCI-IV-2A/multi_subject_json}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export NORM_METHOD="${NORM_METHOD:-z_score}"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
[[ -f "${DATA_PATH}/train.json" ]] || {
    echo "Missing BCI-IV-2a split manifest: ${DATA_PATH}/train.json" >&2
    exit 1
}
exec bash "${REPO_DIR}/scripts/base.sh" "$@"
