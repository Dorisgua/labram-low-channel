#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# PreExp43 AAD baseline: 84 channels, subject-disjoint 24/3/3 split,
# LaBraM mean-pooling classifier, frozen CNN/patch_embed.

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
MASTER_PORT="${MASTER_PORT:-29523}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"
FREEZE_CNN="${FREEZE_CNN:-1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
SEED="${SEED:-0}"

MODEL="labram_base_patch200_200"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/AAD/processed_data_4s_200hz}"
# DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/AAD/processed_data_4s_200hz}"
CHANNEL_SUBSET="${CHANNEL_SUBSET:-aad84}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="${EXP_GROUP:-preexp43_aad84_adabrain_real_subject_split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/aad/${OUTPUT_SCRIPT_NAME}}"
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
if [[ ! -f "${DATA_PATH}/manifest.json" ]]; then
    echo "Missing AAD manifest: ${DATA_PATH}/manifest.json"
    echo "Run: python dataset_maker/make_AAD.py --resume"
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
    --dataset AAD
    --channel_subset "${CHANNEL_SUBSET}"
    --data_path "${DATA_PATH}"
    --sampling_rate 200
    --norm_method z_score
    --completion_scope none
    --pooling_scope low
    --classifier_mode adabrain_all_token
    --classifier_token_scope real
    --best_metric accuracy
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
    --no_auto_resume
)

if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
fi

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n\n'
    echo "AAD protocol: train S0-S23, val S24-S26, test S27-S29; channel_subset=${CHANNEL_SUBSET}"
    echo "Task: binary attended-side classification; 4s windows at 200 Hz"
    echo "Classifier: LaBraM mean pooling; freeze_cnn=${FREEZE_CNN}"
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
