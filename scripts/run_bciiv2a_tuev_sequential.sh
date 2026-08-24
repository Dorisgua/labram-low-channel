#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_FOREGROUND=1
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29650}"

SCRIPTS=(
    "${SCRIPT_DIR}/bciiv2a/A/freeze_cnn.sh"
    "${SCRIPT_DIR}/bciiv2a/N/freeze_cnn.sh"
    "${SCRIPT_DIR}/bciiv2a/N/full_finetune.sh"
    "${SCRIPT_DIR}/bciiv2a/O/freeze_cnn.sh"
    "${SCRIPT_DIR}/bciiv2a/O/full_finetune.sh"
    "${SCRIPT_DIR}/tuev/A/freeze_cnn.sh"
    "${SCRIPT_DIR}/tuev/N/freeze_cnn.sh"
    "${SCRIPT_DIR}/tuev/N/full_finetune.sh"
    "${SCRIPT_DIR}/tuev/O/freeze_cnn.sh"
    "${SCRIPT_DIR}/tuev/O/full_finetune.sh"
)

cd -- "${REPO_DIR}"

echo "仓库：${REPO_DIR}"
echo "GPU：${CUDA_VISIBLE_DEVICES}"
echo "执行顺序：BCI-IV-2a → TUEV"
echo "总脚本数：${#SCRIPTS[@]}"

for script in "${SCRIPTS[@]}"; do
    bash -n "${script}"
done

for ((index = 0; index < ${#SCRIPTS[@]}; index++)); do
    script="${SCRIPTS[index]}"
    relative="${script#"${REPO_DIR}/"}"
    port=$((MASTER_PORT_BASE + index))
    echo
    echo "[$((index + 1))/${#SCRIPTS[@]}] 启动：${relative}"
    echo "端口：${port}"
    MASTER_PORT="${port}" bash "${script}"
    echo "完成：${relative}"
done

echo
echo "BCI-IV-2a 和 TUEV 全部顺序执行完成。"
