#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# PreExp37 EEGMAT base launcher. Defaults to 19 real channels, AdaBrain
# all-token classification, full fine-tuning, and the cross-subject split.

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
MASTER_PORT="${MASTER_PORT:-29520}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"
FREEZE_CNN="${FREEZE_CNN:-0}"

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
export DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/EEGMAT}"
# DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/EEGMAT}"
CHANNEL_SUBSET="${CHANNEL_SUBSET:-eegmat19}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
NORM_METHOD="${NORM_METHOD:-z_score}"
COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
POOLING_SCOPE="${POOLING_SCOPE:-low}"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
DISABLE_REL_POS_BIAS="${DISABLE_REL_POS_BIAS:-0}"
DISABLE_QKV_BIAS="${DISABLE_QKV_BIAS:-0}"
BEST_METRIC="${BEST_METRIC:-accuracy}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="${EXP_GROUP:-preexp37_eegmat19_adabrain_full_finetune_cross_subject}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eegmat/${OUTPUT_SCRIPT_NAME}}"
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
    echo "Missing EEGMAT processed-data directory: ${DATA_PATH}"
    exit 1
fi
if [[ "${COMPLETION_SCOPE}" != "none" && ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing EEGMAT channel prototype: ${CHANNEL_PROTOTYPE_PATH}"
    exit 1
fi
for bool_name in FREEZE_CNN DISABLE_REL_POS_BIAS DISABLE_QKV_BIAS; do
    bool_value="${!bool_name}"
    if [[ "${bool_value}" != "0" && "${bool_value}" != "1" ]]; then
        echo "${bool_name} must be 0 or 1, got: ${bool_value}" >&2
        exit 2
    fi
done

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
    --dataset EEGMAT
    --channel_subset "${CHANNEL_SUBSET}"
    --data_path "${DATA_PATH}"
    --sampling_rate "${SAMPLING_RATE}"
    --norm_method "${NORM_METHOD}"
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
    --abs_pos_emb
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --no_auto_resume
)

if [[ "${COMPLETION_SCOPE}" != "none" ]]; then
    CMD+=(--channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}")
fi
if [[ "${DISABLE_REL_POS_BIAS}" == "1" ]]; then
    CMD+=(--disable_rel_pos_bias)
fi
if [[ "${DISABLE_QKV_BIAS}" == "1" ]]; then
    CMD+=(--disable_qkv_bias)
fi
if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
fi

CMD+=("$@")

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n\n'
    echo "EEGMAT protocol: Subject00-31 train/val, Subject32-35 test; channel_subset=${CHANNEL_SUBSET}"
    echo "Classifier: ${CLASSIFIER_MODE}; token_scope=${CLASSIFIER_TOKEN_SCOPE}; freeze_cnn=${FREEZE_CNN}"
    echo "Completion: ${COMPLETION_SCOPE}; prototype=${CHANNEL_PROTOTYPE_PATH:-<none>}; pooling_scope=${POOLING_SCOPE}"
    echo "LaBraM options: disable_rel_pos_bias=${DISABLE_REL_POS_BIAS}, disable_qkv_bias=${DISABLE_QKV_BIAS}, abs_pos_emb=1"
    echo "Terminal log: ${TERMINAL_LOG}"
} | tee "${TERMINAL_LOG}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    echo "Starting ${RUN_NAME} in foreground"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" 2>&1 | tee -a "${TERMINAL_LOG}"
else
    nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &
    echo "Started ${RUN_NAME} in background"
    echo "PID: $!"
    echo "Log: ${TERMINAL_LOG}"
fi
