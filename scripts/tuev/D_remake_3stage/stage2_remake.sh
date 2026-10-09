#!/usr/bin/env bash
set -euo pipefail

# Dynamic Stage 2 wrapper：加载 Stage 1 corrector，冻结 CNN/corrector，
# 然后复用 TUEV 分类执行器。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-tuev_D_stage2_remake}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"

export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the remake Stage 1 checkpoint-best.pth}"
export STAGE1_CHECKPOINT
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-tuev13}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-tuev13_with_tuev23}"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth}"
# export CORRECTION_SCALE="${CORRECTION_SCALE:-1.0}"

export CLASSIFIER_MODE="mean_pool"
export CLASSIFIER_TOKEN_SCOPE="all"
export FREEZE_CNN="1"
export BEST_METRIC="${BEST_METRIC:-cohen_kappa}"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEED="${SEED:-0}"
export MASTER_PORT="${MASTER_PORT:-auto}"

# 兼容旧 D 脚本的 RUN_BACKGROUND；新接口与 A/N/O 一致，使用 RUN_FOREGROUND。
if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
    case "${RUN_BACKGROUND}" in
        0) export RUN_FOREGROUND=1 ;;
        1) export RUN_FOREGROUND=0 ;;
        *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
    esac
fi

# Stage 1 checkpoint 的 key 不带 student. 前缀，因此必须关闭默认 gzp 过滤。
exec bash "${SCRIPT_DIR}/../base_remake.sh" --model_filter_name "" "$@"
