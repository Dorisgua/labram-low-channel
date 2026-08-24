#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABRAM_BASE_ROOT="${LABRAM_BASE_ROOT:-${PROJECT_ROOT}/../LabraM-Git-Diff}"
export PYTHONPATH="${PROJECT_ROOT}:${LABRAM_BASE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a Stage1 checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for this diagnostic}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CV_SEED="${CV_SEED:-42}"
XGB_SEED="${XGB_SEED:-42}"
UNDERSAMPLE_SEED="${UNDERSAMPLE_SEED:-42}"
BACKGROUND="${BACKGROUND:-1}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/token_vs_pooled_probe.log}"
mkdir -p "${OUTPUT_DIR}"
CMD=("${PYTHON}" -u "${SCRIPT_DIR}/diagnose_token_vs_pooled_probe.py" --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --device "${DEVICE}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --cv-seed "${CV_SEED}" --xgb-seed "${XGB_SEED}" --undersample-seed "${UNDERSAMPLE_SEED}")
if [[ "${BACKGROUND}" == "0" ]]; then
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    exit "${PIPESTATUS[0]}"
fi
nohup setsid "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${OUTPUT_DIR}/token_vs_pooled_probe.pid"
echo "Started token-vs-pooled probe"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
