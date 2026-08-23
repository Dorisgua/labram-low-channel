#!/usr/bin/env bash
set -euo pipefail

# 顺序执行 scripts/ 下除 */others/* 外的所有 Bash 脚本。
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

mapfile -d '' -t SCRIPTS < <(
    find "${SCRIPT_DIR}" -type f -name '*.sh' \
        -not -path '*/others/*' \
        -not -name "${LAUNCHER_NAME}" \
        -print0 | sort -z
)

if [[ "${#SCRIPTS[@]}" -eq 0 ]]; then
    echo "没有找到可执行的 Bash 脚本。" >&2
    exit 1
fi

echo "仓库：${REPO_DIR}"
echo "脚本数量：${#SCRIPTS[@]}"
echo "排除规则：*/others/* 和 ${LAUNCHER_NAME}"
echo "DRY_RUN=${DRY_RUN}"

echo
echo '开始 Bash 语法预检：'
for script in "${SCRIPTS[@]}"; do
    bash -n "${script}"
    echo "  OK  ${script#"${REPO_DIR}/"}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo 'DRY_RUN=1，仅打印执行顺序，不启动训练：'
    for script in "${SCRIPTS[@]}"; do
        echo "  bash ${script#"${REPO_DIR}/"}"
    done
    exit 0
fi

mkdir -p -- "${LOG_ROOT}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAILED=()

echo
echo "开始顺序执行；日志目录：${LOG_ROOT}/${RUN_ID}"
mkdir -p -- "${LOG_ROOT}/${RUN_ID}"

for index in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[index]}"
    relative="${script#"${REPO_DIR}/"}"
    safe_name="${relative//\//__}"
    log_file="${LOG_ROOT}/${RUN_ID}/${safe_name}.log"

    echo
    echo "[$((index + 1))/${#SCRIPTS[@]}] ${relative}"
    echo "日志：${log_file}"

    # if 保证单个脚本失败后继续尝试其余脚本，最后统一汇总失败项。
    if bash "${script}" 2>&1 | tee "${log_file}"; then
        echo "完成：${relative}"
    else
        status="${PIPESTATUS[0]}"
        echo "失败：${relative}（exit=${status}）" >&2
        FAILED+=("${relative} (exit=${status})")
    fi
done

echo
echo '全部脚本处理完毕。'
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "失败数量：${#FAILED[@]}" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi

echo '失败数量：0'
