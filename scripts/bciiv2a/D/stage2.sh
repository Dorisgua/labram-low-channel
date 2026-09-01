#!/usr/bin/env bash
set -euo pipefail

# BCI-IV-2a Dynamic Stage 2: load/freeze the Stage 1 corrector and train
# the same freeze-CNN classification path used by the A/N/O comparison.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# export DATA_PATH="${DATA_PATH:-${REPO_DIR}/preprocessing/BCI-IV-2A/multi_subject_json}"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-bciiv2a_D_stage2}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"

export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/bciiv2a/bciiv2a_D_stage1/checkpoint-best.pth}"
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"

export CHANNEL_SUBSET="bciiv2a13"
export COMPLETION_SCOPE="bciiv2a13_with_bciiv2a22"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_bciiv2a22_cnn_patch_embed_mean.pth}"
export CORRECTION_SCALE="${CORRECTION_SCALE:-0}"

export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
export FREEZE_CNN="1"
# export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"

# export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
# export SEED="${SEED:-0}"
# export MASTER_PORT="${MASTER_PORT:-29563}"

# if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
#     case "${RUN_BACKGROUND}" in
#         0) export RUN_FOREGROUND=1 ;;
#         1) export RUN_FOREGROUND=0 ;;
#         *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
#     esac
# fi

# Stage 1 checkpoint keys are direct backbone keys, without student. prefix.
exec bash "${SCRIPT_DIR}/../base.sh" --model_filter_name "" "$@"
