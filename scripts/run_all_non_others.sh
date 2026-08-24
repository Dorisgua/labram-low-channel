#!/usr/bin/env bash
set -euo pipefail

# 按数据集分阶段执行 O/N/A wrapper：先 bciiv2a，再 erp_core、tuev 等数据集。
# 每个阶段最多同时运行 3 个脚本；前一阶段全部结束后才进入下一阶段。
# 仅创建本调度器不会启动实验；运行本文件才会启动脚本。
#
# 先做：DRY_RUN=1 bash scripts/run_all_non_others.sh
# 实际执行：bash scripts/run_all_non_others.sh
# 指定 GPU：CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_non_others.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER_NAME="$(basename -- "${BASH_SOURCE[0]}")"
DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/outputs/all_non_others_logs}"

cd -- "${REPO_DIR}"

MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT 必须是正整数，当前为：${MAX_CONCURRENT}" >&2
    exit 1
fi

DATASET_GROUPS=(bciiv2a erp_core tuev physionet seed seedv eegmat hgd siena attention aad faced zuo2025)

collect_scripts() {
    local dataset="$1"
    find "${SCRIPT_DIR}/${dataset}" -mindepth 2 -maxdepth 2 -type f -name '*.sh' \
        \( -path '*/O/*' -o -path '*/N/*' -o -path '*/A/*' \) \
        -print0 | sort -z
}

echo "仓库：${REPO_DIR}"
echo "执行阶段：${DATASET_GROUPS[*]}"
echo "每阶段最大并发数：${MAX_CONCURRENT}"
echo "并发 torchrun 端口起点：${MASTER_PORT_BASE}"
echo "扫描规则：仅运行 <dataset>/{O,N,A}/*.sh；不运行根目录旧脚本、*/others/*、base.sh"
echo "DRY_RUN=${DRY_RUN}"

echo
echo '开始 Bash 语法预检：'
TOTAL=0
for dataset in "${DATASET_GROUPS[@]}"; do
    mapfile -d '' -t GROUP_SCRIPTS < <(collect_scripts "${dataset}")
    TOTAL=$((TOTAL + ${#GROUP_SCRIPTS[@]}))
    for script in "${GROUP_SCRIPTS[@]}"; do
        bash -n "${script}"
        echo "  OK  ${script#"${REPO_DIR}/"}"
    done
done

if [[ "${TOTAL}" -eq 0 ]]; then
    echo "没有找到可执行的 Bash 脚本。" >&2
    exit 1
fi
echo "脚本总数：${TOTAL}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo 'DRY_RUN=1，仅打印分阶段执行计划，不启动训练：'
    for dataset in "${DATASET_GROUPS[@]}"; do
        mapfile -d '' -t GROUP_SCRIPTS < <(collect_scripts "${dataset}")
        echo "阶段 ${dataset}（完成本阶段后才进入下一阶段；每批最多 ${MAX_CONCURRENT} 个）："
        for ((index = 0; index < ${#GROUP_SCRIPTS[@]}; index += MAX_CONCURRENT)); do
            echo "  批次 $((index / MAX_CONCURRENT + 1))："
            for ((offset = index; offset < index + MAX_CONCURRENT && offset < ${#GROUP_SCRIPTS[@]}; offset++)); do
                echo "    bash ${GROUP_SCRIPTS[offset]#"${REPO_DIR}/"}"
            done
        done
    done
    exit 0
fi

mkdir -p -- "${LOG_ROOT}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAILED=()
COMPLETED=0

echo
echo "开始分阶段执行；日志目录：${LOG_ROOT}/${RUN_ID}"
mkdir -p -- "${LOG_ROOT}/${RUN_ID}"

for dataset in "${DATASET_GROUPS[@]}"; do
    mapfile -d '' -t GROUP_SCRIPTS < <(collect_scripts "${dataset}")
    echo
    echo "===== 阶段 ${dataset}：${#GROUP_SCRIPTS[@]} 个脚本 ====="

    for ((index = 0; index < ${#GROUP_SCRIPTS[@]}; index += MAX_CONCURRENT)); do
        PIDS=()
        PID_SCRIPTS=()
        echo "启动批次 $((index / MAX_CONCURRENT + 1))（最多 ${MAX_CONCURRENT} 个并发）"

        for ((offset = index; offset < index + MAX_CONCURRENT && offset < ${#GROUP_SCRIPTS[@]}; offset++)); do
            script="${GROUP_SCRIPTS[offset]}"
            relative="${script#"${REPO_DIR}/"}"
            safe_name="${relative//\//__}"
            log_file="${LOG_ROOT}/${RUN_ID}/${safe_name}.log"
            echo "  启动：${relative}"
            echo "  日志：${log_file}"
            # 同一批任务必须使用不同 rendezvous 端口；上一批结束后端口可复用。
            job_port=$((MASTER_PORT_BASE + offset))
            MASTER_PORT="${job_port}" RUN_FOREGROUND=1 bash "${script}" >"${log_file}" 2>&1 &
            PIDS+=("$!")
            PID_SCRIPTS+=("${relative}")
        done

        for ((offset = 0; offset < ${#PIDS[@]}; offset++)); do
            if wait "${PIDS[offset]}"; then
                echo "  完成：${PID_SCRIPTS[offset]}"
            else
                status=$?
                echo "  失败：${PID_SCRIPTS[offset]}（exit=${status}）" >&2
                FAILED+=("${PID_SCRIPTS[offset]} (exit=${status})")
            fi
            COMPLETED=$((COMPLETED + 1))
        done
    done
done

echo
echo "全部脚本处理完毕：${COMPLETED}/${TOTAL}。"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "失败数量：${#FAILED[@]}" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi

echo '失败数量：0'
