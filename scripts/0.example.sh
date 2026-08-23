#!/usr/bin/env bash
set -euo pipefail

# LaBraM-unified-AON 的全数据集公共默认参数入口。
#
# 这个文件只负责“适合跨数据集统一”的运行和优化参数，不负责组装
# run_class_finetuning.py 命令。dataset、data_path、channel_subset、completion、
# prototype、classifier、freeze_cnn、best_metric 和 output_dir 仍由各数据集
# 的 base/wrapper 决定。
#
# 当前无需修改其他脚本即可这样使用：
#   bash scripts/0.example.sh scripts/bciiv2a/33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh
#
# 临时覆盖公共参数：
#   EPOCHS=2 LR=1e-4 DRY_RUN=1 \
#     bash scripts/0.example.sh scripts/seedv/35O.finetune_seedv62_labrambase_freeze_cnn.sh
#
# 也可以由未来的数据集 base source：
#   source "${REPO_DIR}/scripts/0.example.sh"
#
# 注意：只有使用 ${VAR:-default} 读取环境变量的数据集脚本才会接受这里
# 的公共默认值。仍然写死 EPOCHS=...、LR=... 的旧脚本需要在后续迁移时
# 单独参数化；本文件不会悄悄改写它们。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# 1. 跨数据集公共运行参数
# -----------------------------------------------------------------------------

export GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

export TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"
export MODEL="${MODEL:-labram_base_patch200_200}"
export FINETUNE="${FINETUNE:-${REPO_DIR}/checkpoints/labram-base.pth}"

# -----------------------------------------------------------------------------
# 2. 跨数据集公共优化参数
# -----------------------------------------------------------------------------

# 统一基准：单卡 batch 64，训练 50 epochs，学习率 5e-4。
export BATCH_SIZE="${BATCH_SIZE:-64}"
export UPDATE_FREQ="${UPDATE_FREQ:-1}"
export EPOCHS="${EPOCHS:-50}"
export LR="${LR:-5e-4}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export LAYER_DECAY="${LAYER_DECAY:-1.0}"
export DROP_PATH="${DROP_PATH:-0.1}"
export SMOOTHING="${SMOOTHING:-0.1}"
export SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
export SEED="${SEED:-0}"

# 控制变量只提供环境默认值；是否实现由目标数据集脚本决定。
export DRY_RUN="${DRY_RUN:-0}"
export NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"

print_common_config() {
    printf '%s\n' \
        "LaBraM shared defaults:" \
        "  REPO_DIR=${REPO_DIR}" \
        "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
        "  NPROC_PER_NODE=${NPROC_PER_NODE}" \
        "  OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
        "  NUM_WORKERS=${NUM_WORKERS}" \
        "  TORCHRUN=${TORCHRUN}" \
        "  MODEL=${MODEL}" \
        "  FINETUNE=${FINETUNE}" \
        "  BATCH_SIZE=${BATCH_SIZE}" \
        "  UPDATE_FREQ=${UPDATE_FREQ}" \
        "  EPOCHS=${EPOCHS}" \
        "  LR=${LR}" \
        "  WARMUP_EPOCHS=${WARMUP_EPOCHS}" \
        "  WEIGHT_DECAY=${WEIGHT_DECAY}" \
        "  LAYER_DECAY=${LAYER_DECAY}" \
        "  DROP_PATH=${DROP_PATH}" \
        "  SMOOTHING=${SMOOTHING}" \
        "  SAVE_CKPT_FREQ=${SAVE_CKPT_FREQ}" \
        "  SEED=${SEED}" \
        "  DRY_RUN=${DRY_RUN}"
}

# 被 source 时只导出公共变量，不执行任何实验。
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

if [[ $# -eq 0 || "${1:-}" == "--print-config" ]]; then
    print_common_config
    printf '\nUsage:\n  bash %s <dataset-script.sh> [extra arguments...]\n' \
        "${BASH_SOURCE[0]}"
    exit 0
fi

TARGET_SCRIPT="$1"
shift

if [[ "${TARGET_SCRIPT}" != /* ]]; then
    TARGET_SCRIPT="${REPO_DIR}/${TARGET_SCRIPT#./}"
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Missing target script: ${TARGET_SCRIPT}" >&2
    exit 1
fi

print_common_config
printf '  TARGET_SCRIPT=%s\n\n' "${TARGET_SCRIPT}"

cd "${REPO_DIR}"
exec bash "${TARGET_SCRIPT}" "$@"
