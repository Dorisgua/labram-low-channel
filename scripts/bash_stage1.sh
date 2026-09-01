#!/usr/bin/env bash
set -euo pipefail

# Dynamic Stage 1 wrapper。
# 各数据集脚本设置数据与实验变量；这里补齐 Stage 1 默认值和专用 loss，
# torchrun、日志、输出与前后台执行统一复用 scripts/base.sh。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DATASET:?DATASET must be set by the dataset wrapper}"
: "${DATA_PATH:?DATA_PATH must be set by the dataset wrapper}"
: "${CHANNEL_SUBSET:?CHANNEL_SUBSET must be set by the dataset wrapper}"
: "${COMPLETION_SCOPE:?COMPLETION_SCOPE must be set by the dataset wrapper}"
: "${CHANNEL_PROTOTYPE_PATH:?CHANNEL_PROTOTYPE_PATH must be set by the dataset wrapper}"

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-${DATASET,,}_D_stage1}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-run_dynamic_stage1.py}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/${DATASET,,}/dynamic_stage1}"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-20}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export BEST_METRIC="loss"
export POOLING_SCOPE="${POOLING_SCOPE:-low}"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
export FREEZE_CNN="1"
# export DISABLE_REL_POS_BIAS="0"
# export DISABLE_QKV_BIAS="0"
export NO_AUTO_RESUME="1"
export MASTER_PORT="${MASTER_PORT:-29562}"

# 兼容旧 Stage 1 的 RUN_BACKGROUND；统一执行器使用 RUN_FOREGROUND。
if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
    case "${RUN_BACKGROUND}" in
        0) export RUN_FOREGROUND=1 ;;
        1) export RUN_FOREGROUND=0 ;;
        *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
    esac
fi

exec bash "${SCRIPT_DIR}/base.sh" \
    --missing_weight "${MISSING_WEIGHT:-1.0}" \
    --reg_weight "${REG_WEIGHT:-0.01}" \
    --subject_summary_contra_weight "${SUBJECT_SUMMARY_CONTRA_WEIGHT:-0.0}" \
    --task_summary_contra_weight "${TASK_SUMMARY_CONTRA_WEIGHT:-0.0}" \
    --subject_correction_contra_weight "${SUBJECT_CORRECTION_CONTRA_WEIGHT:-0.005}" \
    --task_correction_contra_weight "${TASK_CORRECTION_CONTRA_WEIGHT:-0.005}" \
    --permute_sub_weight "${PERMUTE_SUB_WEIGHT:-5.0}" \
    --permute_task_weight "${PERMUTE_TASK_WEIGHT:-5.0}" \
    --correction_scale "${CORRECTION_SCALE:-1.0}" \
    "$@"
