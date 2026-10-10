#!/usr/bin/env bash
set -euo pipefail

# TUEV Dynamic Stage 1: 13 observed channels -> 23 full channels.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export DATASET="TUEV"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed}"
export CHANNEL_SUBSET="tuev13"
export COMPLETION_SCOPE="tuev13_with_tuev23"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/tuev/D_remake_3stage_stage1/seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)_$$}"

# Match the 17Ah batch and optimization defaults; Stage 1 keeps its own losses.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-50}"
export LR="${LR:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export UPDATE_FREQ="${UPDATE_FREQ:-8}"
export LAYER_DECAY="${LAYER_DECAY:-0.65}"
export SAMPLING_RATE="${SAMPLING_RATE:-200}"
export NORM_METHOD="${NORM_METHOD:-z_score}"
export DISABLE_REL_POS_BIAS="${DISABLE_REL_POS_BIAS:-1}"
export DISABLE_QKV_BIAS="${DISABLE_QKV_BIAS:-1}"

export MISSING_WEIGHT="${MISSING_WEIGHT:-0.0}"
export REG_WEIGHT="${REG_WEIGHT:-0.0}"
export SUBJECT_SUMMARY_CONTRA_WEIGHT="${SUBJECT_SUMMARY_CONTRA_WEIGHT:-0.0}"
export TASK_SUMMARY_CONTRA_WEIGHT="${TASK_SUMMARY_CONTRA_WEIGHT:-0.0}"
export SUBJECT_CORRECTION_CONTRA_WEIGHT="${SUBJECT_CORRECTION_CONTRA_WEIGHT:-1}"
export TASK_CORRECTION_CONTRA_WEIGHT="${TASK_CORRECTION_CONTRA_WEIGHT:-1}"
export PERMUTE_SUB_WEIGHT="${PERMUTE_SUB_WEIGHT:-1.0}"
export PERMUTE_TASK_WEIGHT="${PERMUTE_TASK_WEIGHT:-1.0}"
export SEED="${SEED:-0}"

echo "TUEV Stage 1 output: ${OUTPUT_DIR}"
echo "TUEV Stage 2 checkpoint: ${OUTPUT_DIR%/}/checkpoint-best.pth"
case "${FULLCHANNEL:-1}" in
    1) set -- --fullchannel "$@" ;;
    0) ;;
    *) echo "FULLCHANNEL must be 0 or 1" >&2; exit 2 ;;
esac
exec bash "${REPO_DIR}/scripts/bash_stage1.sh" "$@"
