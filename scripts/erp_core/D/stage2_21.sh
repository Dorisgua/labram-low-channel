#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-erp_core_D_stage2_21}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/erpcore/erp_core_D_stage1_21/checkpoint-best.pth}"
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"

export CHANNEL_SUBSET="erpcore21"
export COMPLETION_SCOPE="erpcore21_with_erpcore28"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
export CORRECTION_SCALE="${CORRECTION_SCALE:-0.02}"

export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
export FREEZE_CNN="1"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-30}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEED="${SEED:-1}"
export MASTER_PORT="${MASTER_PORT:-29562}"

if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
    case "${RUN_BACKGROUND}" in
        0) export RUN_FOREGROUND=1 ;;
        1) export RUN_FOREGROUND=0 ;;
        *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
    esac
fi

exec bash "${SCRIPT_DIR}/../base.sh" --model_filter_name "" "$@"
