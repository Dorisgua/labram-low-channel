#!/usr/bin/env bash
set -euo pipefail

# Run disentanglement, reconstruction, then classification; reuse completed stages.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/outputs/erpcore/D_pipeline_3stage/seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${RUN_ROOT}/stage1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/stage2}"
STAGE1_CHECKPOINT="${STAGE1_OUTPUT_DIR%/}/checkpoint-best.pth"
STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR:-${RUN_ROOT}/stage3}"
STAGE2_CHECKPOINT="${STAGE2_OUTPUT_DIR%/}/checkpoint-best.pth"

if [[ "${STAGE1_OUTPUT_DIR%/}" == "${STAGE2_OUTPUT_DIR%/}" || "${STAGE1_OUTPUT_DIR%/}" == "${STAGE3_OUTPUT_DIR%/}" || "${STAGE2_OUTPUT_DIR%/}" == "${STAGE3_OUTPUT_DIR%/}" ]]; then
    echo "All three stage output directories must differ" >&2
    exit 2
fi

echo "ERP-Core three-stage pipeline root: ${RUN_ROOT}"
echo "Stage 1 output: ${STAGE1_OUTPUT_DIR}"
echo "Stage 2 output: ${STAGE2_OUTPUT_DIR}"
echo "Stage 3 output: ${STAGE3_OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" EPOCHS="${STAGE1_EPOCHS:-50}" RUN_FOREGROUND=1 \
        bash "${SCRIPT_DIR}/stage1.sh" "$@"
    echo "After Stage 1 succeeds, Stage 2 will use: ${STAGE1_CHECKPOINT}"
    echo "Stage 2 reconstruction epochs: ${STAGE2_EPOCHS:-20}"
    echo "Stage 3 loads ${STAGE2_CHECKPOINT}; classification epochs: ${STAGE3_EPOCHS:-50}"
    exit 0
fi

mkdir -p "${RUN_ROOT}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

# Detach the pipeline; all three stages execute sequentially in the child shell.
if [[ "${RUN_BACKGROUND:-1}" == "1" && "${PIPELINE_CHILD:-0}" != "1" ]]; then
    nohup setsid env PIPELINE_CHILD=1 RUN_BACKGROUND=0 \
        RUN_ROOT="${RUN_ROOT}" STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
        STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" STAGE3_OUTPUT_DIR="${STAGE3_OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/stage1_then_stage2.sh" "$@" >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
    echo "Started pipeline in background; PID: $!"
    echo "Log: ${PIPELINE_LOG}"
    exit 0
fi

CHECK_TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"
CHECK_PYTHON="${CHECK_PYTHON:-${CHECK_TORCHRUN%/*}/python}"

check_complete() {
    PYTHONFAULTHANDLER=1 "${CHECK_PYTHON}" "${SCRIPT_DIR}/check_stage_complete.py" "$1" "$2"
}

run_stage() {
    local number="$1" output="$2" epochs="$3"
    shift 3
    if check_complete "${output}" "${epochs}"; then
        touch "${output}/.stage-complete"
        echo "Stage ${number} already complete; skipping"
        return 0
    fi
    # Never overwrite an unfinished experiment when restarting the pipeline.
    if [[ -e "${output}/log.txt" || -e "${output}/checkpoint-best.pth" ]]; then
        echo "Stage ${number} has incomplete or incompatible results in ${output}; stopping. Use a new output directory or explicitly resume this stage." >&2
        return 1
    fi
    mkdir -p "${output}"
    echo "Stage ${number} started: $(date -Is)"
    local status=0
    "$@" || status=$?
    if check_complete "${output}" "${epochs}"; then
        touch "${output}/.stage-complete"
        if [[ "${status}" != 0 ]]; then
            echo "Stage ${number} exited with status ${status}, but completed training and readable weights were verified; continuing"
        fi
        echo "Stage ${number} completed: $(date -Is)"
        return 0
    fi
    echo "Stage ${number} did not complete (exit status ${status}); pipeline stopped" >&2
    return 1
}

run_pipeline() {
    # Prevent two restarts from training in the same directories concurrently.
    exec 9>"${RUN_ROOT}/.pipeline.lock"
    flock -n 9 || { echo "This RUN_ROOT already has an active pipeline" >&2; return 1; }
    run_stage 1 "${STAGE1_OUTPUT_DIR}" "${STAGE1_EPOCHS:-50}" \
        env OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" EPOCHS="${STAGE1_EPOCHS:-50}" RUN_FOREGROUND=1 \
        bash "${SCRIPT_DIR}/stage1.sh" "$@" || return 1
    run_stage 2 "${STAGE2_OUTPUT_DIR}" "${STAGE2_EPOCHS:-20}" \
        env OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" EPOCHS="${STAGE2_EPOCHS:-20}" RUN_FOREGROUND=1 \
        STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}" FINETUNE="${STAGE1_CHECKPOINT}" \
        bash "${SCRIPT_DIR}/stage1_2.sh" || return 1
    run_stage 3 "${STAGE3_OUTPUT_DIR}" "${STAGE3_EPOCHS:-50}" \
        env OUTPUT_DIR="${STAGE3_OUTPUT_DIR}" EPOCHS="${STAGE3_EPOCHS:-50}" RUN_FOREGROUND=1 \
        STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT}" FINETUNE="${STAGE2_CHECKPOINT}" \
        bash "${SCRIPT_DIR}/stage2.sh" || return 1
    echo "All three stages completed"
}

if [[ "${PIPELINE_CHILD:-0}" == "1" ]]; then
    run_pipeline "$@"
else
    run_pipeline "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
fi
