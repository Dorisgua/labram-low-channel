#!/usr/bin/env bash
set -euo pipefail

# Sequential TUEV D run. Stage 2 consumes this run's Stage 1 best checkpoint.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/tuev/D_pipeline/seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${RUN_ROOT}/stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/stage2}"
STAGE1_CHECKPOINT="${STAGE1_OUTPUT_DIR%/}/checkpoint-best.pth"

if [[ "${STAGE1_OUTPUT_DIR%/}" == "${STAGE2_OUTPUT_DIR%/}" ]]; then
    echo "Stage 1 and Stage 2 output directories must differ" >&2
    exit 2
fi

echo "TUEV D pipeline root: ${RUN_ROOT}"
echo "Stage 1 output: ${STAGE1_OUTPUT_DIR}"
echo "Stage 2 output: ${STAGE2_OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" EPOCHS="${STAGE1_EPOCHS:-50}" RUN_FOREGROUND=1 \
        bash "${SCRIPT_DIR}/stage1.sh" "$@"
    echo "After Stage 1 succeeds, Stage 2 will load: ${STAGE1_CHECKPOINT}"
    echo "Stage 2 epochs: ${STAGE2_EPOCHS:-30}"
    exit 0
fi

mkdir -p "${RUN_ROOT}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

if [[ "${RUN_BACKGROUND:-1}" == "1" && "${PIPELINE_CHILD:-0}" != "1" ]]; then
    nohup setsid env PIPELINE_CHILD=1 RUN_BACKGROUND=0 \
        RUN_ROOT="${RUN_ROOT}" STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
        STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/stage1_then_stage2.sh" "$@" > "${PIPELINE_LOG}" 2>&1 < /dev/null &
    echo "Started pipeline in background; PID: $!"
    echo "Log: ${PIPELINE_LOG}"
    exit 0
fi

run_pipeline() {
    echo "Stage 1 started: $(date -Is)"
    OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" EPOCHS="${STAGE1_EPOCHS:-50}" RUN_FOREGROUND=1 \
        bash "${SCRIPT_DIR}/stage1.sh" "$@"

    if [[ ! -s "${STAGE1_CHECKPOINT}" ]]; then
        echo "Missing Stage 1 best checkpoint: ${STAGE1_CHECKPOINT}" >&2
        exit 1
    fi
    echo "Stage 1 completed: $(date -Is)"
    echo "Stage 2 loading: ${STAGE1_CHECKPOINT}"

    OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" EPOCHS="${STAGE2_EPOCHS:-30}" RUN_FOREGROUND=1 \
        STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}" FINETUNE="${STAGE1_CHECKPOINT}" \
        bash "${SCRIPT_DIR}/stage2.sh"
    echo "Stage 2 completed: $(date -Is)"
}

if [[ "${PIPELINE_CHILD:-0}" == "1" ]]; then
    run_pipeline "$@"
else
    run_pipeline "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
fi
