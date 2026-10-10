#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ===== 只改这里：GPU、seed，以及要跑的 sh（从上到下依次运行） =====
GPU_ID="${GPU_ID:-0}"
SCRIPTS=(
    # "seed=0 bciiv2a/A/cbramod_freeze_cnn.sh"
    # "seed=0 tuev/O/freeze_cnn_remake.sh"
    "seed=1 bciiv2a/A/cbramod_freeze_cnn.sh"
    "seed=2 bciiv2a/A/cbramod_freeze_cnn.sh"
    "seed=1 bciiv2a/O/cbramod_freeze_cnn.sh"
    "seed=2 bciiv2a/O/cbramod_freeze_cnn.sh"
    "seed=1 bciiv2a/N/cbramod_freeze_cnn.sh"
    "seed=2 bciiv2a/N/cbramod_freeze_cnn.sh"
    "seed=1 bciiv2a/D_3stage/cbramod_stage1_then_stage2.sh"
    "seed=2 bciiv2a/D_3stage/cbramod_stage1_then_stage2.sh"

    "seed=1 shu/D/stage1_then_stage2.sh"
    "seed=2 shu/D/stage1_then_stage2.sh"

    "seed=0 shu/D_3stage/stage1_then_stage2.sh"
    "seed=1 shu/D_3stage/stage1_then_stage2.sh"
    "seed=2 shu/D_3stage/stage1_then_stage2.sh"

    "seed=0 erp_core/D_3stage/stage1_then_stage2.sh"
    "seed=1 erp_core/D_3stage/stage1_then_stage2.sh"
    "seed=2 erp_core/D_3stage/stage1_then_stage2.sh"

    "seed=0 tuev/D_remake_3stage/stage1_then_stage2_remake.sh"
    "seed=1 tuev/D_remake_3stage/stage1_then_stage2_remake.sh"
    "seed=2 tuev/D_remake_3stage/stage1_then_stage2_remake.sh"
)
# 不想跑某项，就删除或注释对应行。
# ==============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}" GPU_IDS="${GPU_ID}"
# 确保每项完整结束后才启动下一项；D 内部也顺序执行两个阶段。
export RUN_FOREGROUND=1 RUN_BACKGROUND=0 NPROC_PER_NODE=1

# 先检查所有任务，避免跑到一半才发现配置错误。
for task in "${SCRIPTS[@]}"; do
    read -r seed_spec script extra <<< "${task}"
    if [[ ! "${seed_spec}" =~ ^seed=[0-9]+$ || -z "${script}" || -n "${extra}" ]]; then
        echo "任务格式错误（应为 seed=0 路径）：${task}" >&2
        exit 1
    fi
    [[ -f "${SCRIPT_DIR}/${script}" ]] || {
        echo "找不到脚本：${SCRIPT_DIR}/${script}" >&2
        exit 1
    }
done

# 整个队列在后台运行；子任务保持前台，确保严格串行。
# SERIAL_BACKGROUND=0 bash 本脚本 可改为前台运行。
if [[ "${SERIAL_BACKGROUND:-1}" == "1" ]]; then
    REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
    QUEUE_LOG_DIR="${REPO_DIR}/outputs/tuev/serial_remake_logs"
    mkdir -p "${QUEUE_LOG_DIR}"
    QUEUE_LOG="${QUEUE_LOG_DIR}/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
    nohup env SERIAL_BACKGROUND=0 GPU_ID="${GPU_ID}" \
        bash "${SCRIPT_DIR}/run_serial.sh" > "${QUEUE_LOG}" 2>&1 < /dev/null &
    QUEUE_PID=$!
    echo "后台队列已启动，PID：${QUEUE_PID}"
    echo "总日志：${QUEUE_LOG}"
    echo "查看进度：tail -f '${QUEUE_LOG}'"
    exit 0
fi

for task in "${SCRIPTS[@]}"; do
    read -r seed_spec script <<< "${task}"
    task_seed="${seed_spec#seed=}"
    echo "开始：${script}，GPU ${GPU_ID}，seed ${task_seed}"
    SEED="${task_seed}" bash "${SCRIPT_DIR}/${script}"
    echo "完成：${script}"
done

echo "全部任务完成。"
