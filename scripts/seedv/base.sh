#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export DATASET="SEEDV"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-labram}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export UPDATE_FREQ="${UPDATE_FREQ:-8}"
export LAYER_DECAY="${LAYER_DECAY:-0.65}"
export NORM_METHOD="${NORM_METHOD:-z_score}"
export BEST_METRIC="${BEST_METRIC:-accuracy}"
exec bash "${REPO_DIR}/scripts/base.sh" "$@"
