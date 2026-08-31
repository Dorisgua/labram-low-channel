#!/usr/bin/env bash
set -euo pipefail

# 公共 LaBraM 训练执行器。
# 数据集 base 负责设置 DATASET/DATA_PATH 等数据集变量；wrapper 负责设置
# 训练入口、O/N/A/D、completion、prototype、classifier 和 freeze/full 等实验变量。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"

: "${DATASET:?DATASET must be set by the dataset base}"
: "${DATA_PATH:?DATA_PATH must be set by the dataset base}"
: "${CHANNEL_SUBSET:?CHANNEL_SUBSET must be set by the wrapper or dataset base}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TORCHRUN="${TORCHRUN:-${REPO_DIR}/../../micromamba-root/envs/labram/bin/torchrun}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-run_class_finetuning.py}"
MODEL="${MODEL:-labram_base_patch200_200}"
FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"

BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-1.0}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
NORM_METHOD="${NORM_METHOD:-z_score}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
SEED="${SEED:-0}"

COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
POOLING_SCOPE="${POOLING_SCOPE:-low}"
CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
CORRECTION_SCALE="${CORRECTION_SCALE:-1.0}"
BEST_METRIC="${BEST_METRIC:-accuracy}"
FREEZE_CNN="${FREEZE_CNN:-1}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_REL_POS_BIAS="${DISABLE_REL_POS_BIAS:-0}"
DISABLE_QKV_BIAS="${DISABLE_QKV_BIAS:-0}"

RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-${OUTPUT_SCRIPT_NAME}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
RUN_NAME="${RUN_PREFIX}${RUN_TAG:+_${RUN_TAG}}_${RUN_ID}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    # Dynamic Stage 1 keeps checkpoint-best.pth at a stable handoff path.
    OUTPUT_DIR="${OUTPUT_DIR%/}/"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_DIR%/}}"
    TB_LOG_DIR="${TB_LOG_DIR:-${OUTPUT_DIR}tensorboard/}"
    TERMINAL_LOG_DIR="${TERMINAL_LOG_DIR:-${OUTPUT_DIR}run_logs}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/${DATASET,,}/${OUTPUT_SCRIPT_NAME}}"
    OUTPUT_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}/"
    TB_LOG_DIR="${OUTPUT_ROOT}/tensorboard/${RUN_NAME}/"
    TERMINAL_LOG_DIR="${OUTPUT_ROOT}/run_logs"
fi
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${RUN_NAME}.log"

if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing torchrun executable: ${TORCHRUN}" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_ENTRYPOINT}" ]]; then
    echo "Missing training entrypoint: ${TRAIN_ENTRYPOINT}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" && ! -d "${DATA_PATH}" ]]; then
    echo "Missing dataset path: ${DATA_PATH}" >&2
    exit 1
fi
if [[ ! -f "${FINETUNE}" ]]; then
    echo "Missing finetune checkpoint: ${FINETUNE}" >&2
    exit 1
fi
if [[ "${COMPLETION_SCOPE}" != "none" && ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing channel prototype file: ${CHANNEL_PROTOTYPE_PATH}" >&2
    exit 1
fi

CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    "${TRAIN_ENTRYPOINT}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${TB_LOG_DIR}"
    --model "${MODEL}"
    --finetune "${FINETUNE}"
    --dataset "${DATASET}"
    --channel_subset "${CHANNEL_SUBSET}"
    --data_path "${DATA_PATH}"
    --sampling_rate "${SAMPLING_RATE}"
    --norm_method "${NORM_METHOD}"
    --completion_scope "${COMPLETION_SCOPE}"
    --pooling_scope "${POOLING_SCOPE}"
    --classifier_mode "${CLASSIFIER_MODE}"
    --classifier_token_scope "${CLASSIFIER_TOKEN_SCOPE}"
    --correction_scale "${CORRECTION_SCALE}"
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
)

if [[ "${DISABLE_REL_POS_BIAS}" == "1" ]]; then CMD+=(--disable_rel_pos_bias); fi
if [[ "${DISABLE_QKV_BIAS}" == "1" ]]; then CMD+=(--disable_qkv_bias); fi

if [[ -n "${CHANNEL_PROTOTYPE_PATH}" && "${COMPLETION_SCOPE}" != "none" ]]; then
    CMD+=(--channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}")
fi
if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
elif [[ "${FREEZE_CNN}" != "0" ]]; then
    echo "FREEZE_CNN must be 0 or 1, got: ${FREEZE_CNN}" >&2
    exit 2
fi
if [[ -n "${RESUME}" ]]; then CMD+=(--resume "${RESUME}"); fi
if [[ "${EVAL_ONLY}" == "1" ]]; then CMD+=(--eval); fi
if [[ "${NO_AUTO_RESUME}" == "1" ]]; then CMD+=(--no_auto_resume); fi
CMD+=("$@")

print_cmd() {
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
}

if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd
    exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${TB_LOG_DIR}" "${TERMINAL_LOG_DIR}"
{
    echo "Command:"
    print_cmd
    echo "Output root: ${OUTPUT_ROOT}"
} > "${TERMINAL_LOG}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" 2>&1 | tee -a "${TERMINAL_LOG}"
else
    nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &
    echo "Started ${RUN_NAME}"
    echo "PID: $!"
    echo "Log: ${TERMINAL_LOG}"
fi
