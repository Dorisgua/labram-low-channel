#!/usr/bin/env bash
set -euo pipefail

# Dynamic Stage 2：只输入 x_obs=12 导联，使用已经训练好的 Stage 1 corrector，
# 冻结 corrector 后再微调 Transformer 和分类头。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

TORCHRUN="${TORCHRUN:-${REPO_DIR}/../../micromamba-root/envs/labram/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0}"
MASTER_PORT="${MASTER_PORT:-29561}"
DATA_PATH="${DATA_PATH:-/inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation/simple_data.pt}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/erp_core/dynamic_stage1/checkpoint-best.pth}"
PROTOTYPE="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/erp_core/dynamic_stage2}"

# run_class_finetuning.py 已注册 Dynamic 模型，下面命令直接启动 Stage 2。
CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node=1
    --master_port="${MASTER_PORT}"
    run_class_finetuning.py
    --model labram_dynamic_base_patch200_200
    --dataset ERPCORE
    --data_path "${DATA_PATH}"
    --channel_subset erpcore12
    --completion_scope erpcore12_with_erpcore28
    --channel_prototype_path "${PROTOTYPE}"
    --finetune "${STAGE1_CHECKPOINT}"
    --model_filter_name ""
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${OUTPUT_DIR}/tensorboard"
    # --classifier_mode mean_pool
    --batch_size "${BATCH_SIZE:-64}"
    --epochs "${EPOCHS:-30}"
    --update_freq "${UPDATE_FREQ:-1}"
    --layer_decay "${LAYER_DECAY:-1.0}"
    --lr "${LR:-5e-4}"
    --weight_decay "${WEIGHT_DECAY:-0.05}"
    --num_workers "${NUM_WORKERS:-4}"
    --seed "${SEED:-1}"
    --freeze_cnn
    --disable_rel_pos_bias
    --disable_qkv_bias
    --classifier_mode adabrain_all_token
    --classifier_token_scope all
    --no_auto_resume
)

mkdir -p "${OUTPUT_DIR}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

if [[ "${RUN_BACKGROUND}" == "1" ]]; then
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/stage2_${RUN_ID}.log}"
    PID_FILE="${PID_FILE:-${OUTPUT_DIR}/stage2_${RUN_ID}.pid}"

    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        "${CMD[@]}" "$@" >"${RUN_LOG}" 2>&1 &
    STAGE2_PID=$!
    printf '%s\n' "${STAGE2_PID}" >"${PID_FILE}"

    echo "Dynamic Stage 2 started in background: PID=${STAGE2_PID}"
    echo "Log: ${RUN_LOG}"
    echo "PID file: ${PID_FILE}"
else
    # 调试时可以设置 RUN_BACKGROUND=0，恢复前台执行。
    env CUDA_VISIBLE_DEVICES="${GPU_IDS}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        "${CMD[@]}" "$@"
fi
