#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# PreExp38 minimal Zuo2025 baseline: cue-relative 0-4 s, 30 channels,
# no completion, frozen CNN, LaBraM mean-pooling classifier.

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
MASTER_PORT="${MASTER_PORT:-29518}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-2}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
SEED="${SEED:-0}"

MODEL="labram_base_patch200_200"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Zuo2025/processed_data_4s_200hz}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="${EXP_GROUP:-preexp38_zuo2025_30_adabrain_real_cross_subject}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/zuo2025/${OUTPUT_SCRIPT_NAME}}"
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
    echo "Missing Zuo2025 4-second manifest: ${DATA_PATH}/manifest.json"
    echo "Run: python dataset_maker/make_Zuo2025.py"
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
    --dataset Zuo2025
    --channel_subset zuo2025_30
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
    --freeze_cnn
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --no_auto_resume
)

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n\n'
    echo "Zuo2025 window: cue-relative 0-4 s; 200 Hz; 30 EEG channels"
    echo "Protocol: subjects 1-24 train, 25-27 validation, 28-30 test"
    echo "Classifier: LaBraM mean pooling; CNN/patch_embed frozen"
    echo "Terminal log: ${TERMINAL_LOG}"
} | tee "${TERMINAL_LOG}"

"${CMD[@]}" 2>&1 | tee -a "${TERMINAL_LOG}"
