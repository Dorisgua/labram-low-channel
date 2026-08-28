#!/usr/bin/env bash
set -euo pipefail

# Dynamic Stage 1 公共执行器。
# 各数据集脚本只负责设置 DATASET、DATA_PATH、CHANNEL_SUBSET 等变量，
# 具体训练命令统一在这里维护。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${DATASET:?DATASET must be set by the dataset wrapper}"
: "${DATA_PATH:?DATA_PATH must be set by the dataset wrapper}"
: "${CHANNEL_SUBSET:?CHANNEL_SUBSET must be set by the dataset wrapper}"
: "${COMPLETION_SCOPE:?COMPLETION_SCOPE must be set by the dataset wrapper}"
: "${CHANNEL_PROTOTYPE_PATH:?CHANNEL_PROTOTYPE_PATH must be set by the dataset wrapper}"

TORCHRUN="${TORCHRUN:-${REPO_DIR}/../../micromamba-root/envs/labram/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0}"
MASTER_PORT="${MASTER_PORT:-29562}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/${DATASET,,}/dynamic_stage1}"

[[ -x "${TORCHRUN}" ]] || {
    echo "Missing torchrun executable: ${TORCHRUN}" >&2
    exit 1
}
[[ -f "${DATA_PATH}" || -d "${DATA_PATH}" ]] || {
    echo "Missing dataset path: ${DATA_PATH}" >&2
    exit 1
}
[[ -f "${FINETUNE}" ]] || {
    echo "Missing finetune checkpoint: ${FINETUNE}" >&2
    exit 1
}
[[ -f "${CHANNEL_PROTOTYPE_PATH}" ]] || {
    echo "Missing channel prototype checkpoint: ${CHANNEL_PROTOTYPE_PATH}" >&2
    exit 1
}

CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE:-1}"
    --master_port="${MASTER_PORT}"
    run_dynamic_stage1.py
    --model "${MODEL:-labram_dynamic_base_patch200_200}"
    --dataset "${DATASET}"
    --data_path "${DATA_PATH}"
    --channel_subset "${CHANNEL_SUBSET}"
    --completion_scope "${COMPLETION_SCOPE}"
    --channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}"
    --finetune "${FINETUNE}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${OUTPUT_DIR}/tensorboard"
    --batch_size "${BATCH_SIZE:-64}"
    --epochs "${EPOCHS:-20}"
    --update_freq "${UPDATE_FREQ:-1}"
    --lr "${LR:-5e-4}"
    --weight_decay "${WEIGHT_DECAY:-0.05}"
    --warmup_epochs "${WARMUP_EPOCHS:-5}"
    --missing_weight "${MISSING_WEIGHT:-1.0}"
    --reg_weight "${REG_WEIGHT:-0.01}"
    --subject_summary_contra_weight "${SUBJECT_SUMMARY_CONTRA_WEIGHT:-0.0}"
    --task_summary_contra_weight "${TASK_SUMMARY_CONTRA_WEIGHT:-0.0}"
    --subject_correction_contra_weight "${SUBJECT_CORRECTION_CONTRA_WEIGHT:-0.005}"
    --task_correction_contra_weight "${TASK_CORRECTION_CONTRA_WEIGHT:-0.005}"
    --permute_sub_weight "${PERMUTE_SUB_WEIGHT:-5.0}"
    --permute_task_weight "${PERMUTE_TASK_WEIGHT:-5.0}"
    --correction_scale "${CORRECTION_SCALE:-1.0}"
    --sampling_rate "${SAMPLING_RATE:-200}"
    --norm_method "${NORM_METHOD:-z_score}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED:-0}"
    --disable_rel_pos_bias
    --disable_qkv_bias
    --no_auto_resume
)

mkdir -p "${OUTPUT_DIR}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

if [[ "${RUN_BACKGROUND}" == "1" ]]; then
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/stage1_${RUN_ID}.log}"
    PID_FILE="${PID_FILE:-${OUTPUT_DIR}/stage1_${RUN_ID}.pid}"

    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        "${CMD[@]}" "$@" >"${RUN_LOG}" 2>&1 &
    STAGE1_PID=$!
    printf '%s\n' "${STAGE1_PID}" >"${PID_FILE}"

    echo "Dynamic Stage 1 started in background: PID=${STAGE1_PID}"
    echo "Log: ${RUN_LOG}"
    echo "PID file: ${PID_FILE}"
else
    # 原前台执行方式，设置 RUN_BACKGROUND=0 时使用。
    env CUDA_VISIBLE_DEVICES="${GPU_IDS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        "${CMD[@]}" "$@"
fi
