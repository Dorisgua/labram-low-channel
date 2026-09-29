#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export DATASET="SEED"
export DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED/processed_data}"
# export DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/SEED/processed_data}"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-seed_D_stage2}"
export MODEL="labram_dynamic_base_patch200_200"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the Stage 1 checkpoint-best.pth for this run}"
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"
export CHANNEL_SUBSET="seed23"
export COMPLETION_SCOPE="seed23_with_seed62"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_seed62_cnn_patch_embed_mean.pth}"
export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
export FREEZE_CNN="1"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-30}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEED="${SEED:-0}"
export MASTER_PORT="${MASTER_PORT:-auto}"

exec bash "${REPO_DIR}/scripts/base.sh" --model_filter_name "" "$@"
