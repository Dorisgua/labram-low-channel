#!/usr/bin/env bash
set -euo pipefail

# BCI-IV-2a Dynamic Stage 1: learn 13 -> 22 channel latent completion.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export DATASET="bciiv2a"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/preprocessing/BCI-IV-2A/multi_subject_json}"
export CHANNEL_SUBSET="bciiv2a13"
export COMPLETION_SCOPE="bciiv2a13_with_bciiv2a22"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_bciiv2a22_cnn_patch_embed_mean.pth}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/bciiv2a/bciiv2a_D_stage1/p_miss_add_delta_f1smooth_missingw2_epoch50}"

# Match ERP-Core D Stage 1 loss defaults; BCI uses four temporal patches.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-50}"
export LR="${LR:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export SAMPLING_RATE="${SAMPLING_RATE:-20}"
export NORM_METHOD="${NORM_METHOD:-z_score}"

export MISSING_WEIGHT="${MISSING_WEIGHT:-2.0}"
export REG_WEIGHT="${REG_WEIGHT:-0.001}"
export SUBJECT_SUMMARY_CONTRA_WEIGHT="${SUBJECT_SUMMARY_CONTRA_WEIGHT:-0}"
export TASK_SUMMARY_CONTRA_WEIGHT="${TASK_SUMMARY_CONTRA_WEIGHT:-0.0}"
export SUBJECT_CORRECTION_CONTRA_WEIGHT="${SUBJECT_CORRECTION_CONTRA_WEIGHT:-0.02}"
export TASK_CORRECTION_CONTRA_WEIGHT="${TASK_CORRECTION_CONTRA_WEIGHT:-0}"
export PERMUTE_SUB_WEIGHT="${PERMUTE_SUB_WEIGHT:-0}"
export PERMUTE_TASK_WEIGHT="${PERMUTE_TASK_WEIGHT:-0}"
export CORRECTION_SCALE="${CORRECTION_SCALE:-0.02}"
export SEED="${SEED:-0}"

exec bash "${REPO_DIR}/scripts/bash_stage1.sh" "$@"
