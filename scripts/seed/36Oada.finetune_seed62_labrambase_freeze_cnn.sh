#!/usr/bin/env bash
set -euo pipefail
# SEED cross-subject base launcher. Defaults to all 62 real channels,
# frozen CNN/patch_embed, and the AdaBrain all-token classification head.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-}"
RUN_NAME="${RUN_PREFIX}${RUN_TAG:+_${RUN_TAG}}_${RUN_ID}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29510}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
FREEZE_CNN="${FREEZE_CNN:-1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
SEED="${SEED:-0}"

MODEL="labram_base_patch200_200"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATASET="SEED"
CHANNEL_SUBSET="${CHANNEL_SUBSET:-seed62}"
DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED/processed_data}"
BEST_METRIC="${BEST_METRIC:-accuracy}"

COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
POOLING_SCOPE="${POOLING_SCOPE:-low}"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"

RESUME="${RESUME:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"

TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="${EXP_GROUP:-preexp36_seed_cross_subject}"
OUTPUT_ROOT="./outputs/seed/${EXP_GROUP}"
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
if [[ ! -d "${DATA_PATH}" ]]; then
    echo "Missing SEED processed-data directory: ${DATA_PATH}"
    exit 1
fi
if [[ "${COMPLETION_SCOPE}" != "none" && ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing SEED channel prototype: ${CHANNEL_PROTOTYPE_PATH}"
    echo "Generate it with: python docs/prototypes/01_generate_seed_cnn_patch_prototypes.py"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${TB_LOG_DIR}" "${TERMINAL_LOG_DIR}"

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
    --classifier_mode "${CLASSIFIER_MODE}"
    --classifier_token_scope "${CLASSIFIER_TOKEN_SCOPE}"
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

if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
elif [[ "${FREEZE_CNN}" != "0" ]]; then
    echo "FREEZE_CNN must be 0 or 1, got: ${FREEZE_CNN}" >&2
    exit 2
fi

if [[ "${COMPLETION_SCOPE}" != "none" ]]; then
    CMD+=(--channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}")
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

CMD+=("$@")

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    echo
    echo "Output:"
} > "${TERMINAL_LOG}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    echo "Starting ${RUN_NAME} in foreground"
    echo "Log: ${TERMINAL_LOG}"
    echo "Global batch size: $((BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE))"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" 2>&1 | tee -a "${TERMINAL_LOG}"
else
    nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &

    echo "Started ${RUN_NAME}"
    echo "Log: ${TERMINAL_LOG}"
    echo "Global batch size: $((BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE))"
fi
