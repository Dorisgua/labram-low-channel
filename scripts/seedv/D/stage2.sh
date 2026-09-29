#!/usr/bin/env bash
set -euo pipefail

# SEED-V Dynamic Stage 2: load Stage 1 corrector and train classifier.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-seedv_D_stage2}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-labram}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the Stage 1 checkpoint-best.pth for this run}"
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"

export CHANNEL_SUBSET="seedv23"
export COMPLETION_SCOPE="seedv23_with_seedv62"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth}"
export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
export FREEZE_CNN="1"
export BEST_METRIC="${BEST_METRIC:-accuracy}"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-30}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEED="${SEED:-0}"
export MASTER_PORT="${MASTER_PORT:-auto}"

if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
    case "${RUN_BACKGROUND}" in
        0) export RUN_FOREGROUND=1 ;;
        1) export RUN_FOREGROUND=0 ;;
        *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
    esac
fi

exec bash "${SCRIPT_DIR}/../base.sh" --model_filter_name "" "$@"
