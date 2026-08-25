#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# SEED-V 62-channel launcher. By default it freezes patch_embed/CNN and trains
# the Transformer plus the original LaBraM mean-pooling classification head.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_PREFIX}_${RUN_ID}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29510}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-8}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
SEED="${SEED:-0}"

MODEL="${MODEL:-labram_base_patch200_200}"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATASET="SEEDV"
CHANNEL_SUBSET="${CHANNEL_SUBSET:-seedv62}"
DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/SEED_V/SEED-V-labram}"
# DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-labram}"
BEST_METRIC="${BEST_METRIC:-accuracy}"

COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
POOLING_SCOPE="${POOLING_SCOPE:-low}"
FREEZE_CNN="${FREEZE_CNN:-1}"

RESUME="${RESUME:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="${EXP_GROUP:-preexp35_seedv62_freeze_cnn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/seedv/${OUTPUT_SCRIPT_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}/"
TB_LOG_DIR="${OUTPUT_ROOT}/tensorboard/${RUN_NAME}/"
TERMINAL_LOG_DIR="${OUTPUT_ROOT}/run_logs"
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${RUN_NAME}.log"

if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing torchrun executable: ${TORCHRUN}"
    exit 1
fi
if [[ ! -f "${FINETUNE}" ]]; then
    echo "Missing finetune checkpoint: ${FINETUNE}"
    exit 1
fi
if [[ ! -f "${DATA_PATH}/metadata.csv" ]]; then
    echo "Missing SEED-V metadata: ${DATA_PATH}/metadata.csv"
    exit 1
fi
for split in processed_train processed_eval processed_test; do
    if [[ ! -d "${DATA_PATH}/${split}" ]]; then
        echo "Missing SEED-V split directory: ${DATA_PATH}/${split}"
        exit 1
    fi
done
if [[ "${COMPLETION_SCOPE}" != "none" && ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing SEED-V channel prototype: ${CHANNEL_PROTOTYPE_PATH}"
    exit 1
fi

CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    run_class_finetuning.py
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${TB_LOG_DIR}"
    --model "${MODEL}"
    --finetune "${FINETUNE}"
    --dataset "${DATASET}"
    --channel_subset "${CHANNEL_SUBSET}"
    --data_path "${DATA_PATH}"
    --completion_scope "${COMPLETION_SCOPE}"
    --pooling_scope "${POOLING_SCOPE}"
    --classifier_mode adabrain_all_token
    --classifier_token_scope real
    --best_metric "${BEST_METRIC}"
    --batch_size "${BATCH_SIZE}"
    --update_freq "${UPDATE_FREQ}"
    --lr "${LR}"
    --epochs "${EPOCHS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --weight_decay "${WEIGHT_DECAY}"
    --layer_decay "${LAYER_DECAY}"
    --drop_path "${DROP_PATH}"
    --smoothing "${SMOOTHING}"
    --save_ckpt_freq "${SAVE_CKPT_FREQ}"
    --disable_rel_pos_bias
    --abs_pos_emb
    --disable_qkv_bias
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
)

if [[ -n "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    CMD+=(--channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}")
fi
if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
fi
if [[ -n "${RESUME}" ]]; then
    CMD+=(--resume "${RESUME}")
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
    CMD+=(--eval)
fi
if [[ "${NO_AUTO_RESUME}" == "1" ]]; then
    CMD+=(--no_auto_resume)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${TB_LOG_DIR}" "${TERMINAL_LOG_DIR}"
{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n\nOutput:\n'
} > "${TERMINAL_LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &

echo "Started ${RUN_NAME}"
echo "Log: ${TERMINAL_LOG}"
echo "Global batch size: $((BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE))"
