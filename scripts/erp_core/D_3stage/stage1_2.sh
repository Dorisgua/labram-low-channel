#!/usr/bin/env bash
set -euo pipefail
# Continue training the corrector from Stage 1, using reconstruction plus correction contrast and swaps.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the disentanglement checkpoint-best.pth}"
export FINETUNE="${STAGE1_CHECKPOINT}"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/erpcore/D_3stage_stage2/seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)_$$}"
export MISSING_WEIGHT="${RECON_MISSING_WEIGHT:-50000.0}"
export FULLCHANNEL=0
export EPOCHS="${EPOCHS:-20}"
export REG_WEIGHT=0 SUBJECT_SUMMARY_CONTRA_WEIGHT=0 TASK_SUMMARY_CONTRA_WEIGHT=0
export SUBJECT_CORRECTION_CONTRA_WEIGHT=1 TASK_CORRECTION_CONTRA_WEIGHT=1
export PERMUTE_SUB_WEIGHT=1 PERMUTE_TASK_WEIGHT=1
exec bash "${SCRIPT_DIR}/stage1.sh" --model_filter_name "" "$@"
