#!/usr/bin/env bash
set -euo pipefail

# ERP CORE Dynamic Stage 1 wrapper。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export DATASET="ERPCORE"
export DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation/simple_data.pt}"
export CHANNEL_SUBSET="erpcore12"
export COMPLETION_SCOPE="erpcore12_with_erpcore28"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/erpcore/erp_core_D_stage1}"

# ERP CORE Dynamic Stage 1：参考 preexp16 的训练与 loss 配置。
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-20}"
export LR="${LR:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export NORM_METHOD="${NORM_METHOD:-z_score}"

export MISSING_WEIGHT="${MISSING_WEIGHT:-20.0}"
export REG_WEIGHT="${REG_WEIGHT:-0.001}"
export SUBJECT_SUMMARY_CONTRA_WEIGHT="${SUBJECT_SUMMARY_CONTRA_WEIGHT:-0.0}"
export TASK_SUMMARY_CONTRA_WEIGHT="${TASK_SUMMARY_CONTRA_WEIGHT:-0.0}"
export SUBJECT_CORRECTION_CONTRA_WEIGHT="${SUBJECT_CORRECTION_CONTRA_WEIGHT:-0.005}"
export TASK_CORRECTION_CONTRA_WEIGHT="${TASK_CORRECTION_CONTRA_WEIGHT:-0.005}"
export PERMUTE_SUB_WEIGHT="${PERMUTE_SUB_WEIGHT:-5.0}"
export PERMUTE_TASK_WEIGHT="${PERMUTE_TASK_WEIGHT:-5.0}"
export CORRECTION_SCALE="${CORRECTION_SCALE:-0.02}"
export SEED="${SEED:-0}"

# Stage 1 保留 reconstruction 专用 epoch/loss，只对齐 A/N/O 的公共模型配置。
exec bash "${REPO_DIR}/scripts/bash_stage1.sh" "$@"
