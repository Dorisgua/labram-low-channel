#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

run_worker() {
    local master_dir="$1"
    local active_pid=""

    stop_active_experiment() {
        if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
            kill "${active_pid}" 2>/dev/null || true
            wait "${active_pid}" 2>/dev/null || true
        fi
        echo "ABC master stopped at $(date --iso-8601=seconds)"
        exit 143
    }
    trap stop_active_experiment TERM INT

    run_experiment() {
        local experiment_id="$1"
        local experiment_name="$2"
        local wrapper="$3"
        local exit_code

        echo
        echo "===== START ${experiment_id}: ${experiment_name} ====="
        echo "Start time: $(date --iso-8601=seconds)"
        echo "Run directory is reported by the launcher below."

        BACKGROUND=0 bash "${wrapper}" &
        active_pid=$!
        set +e
        wait "${active_pid}"
        exit_code=$?
        set -e
        active_pid=""

        echo "Finish time: $(date --iso-8601=seconds)"
        echo "Exit code: ${exit_code}"
        if [[ "${exit_code}" -ne 0 ]]; then
            echo "FAILED AT EXPERIMENT ${experiment_id}"
            return "${exit_code}"
        fi
        echo "===== FINISH ${experiment_id}: ${experiment_name} ====="
    }

    echo "ABC four-loss master started at $(date --iso-8601=seconds)"
    echo "Master directory: ${master_dir}"
    echo "Common configuration:"
    echo "  SEED=${SEED}"
    echo "  EPOCHS=${EPOCHS}"
    echo "  BATCH_SIZE=${BATCH_SIZE}"
    echo "  LR=${LR}"
    echo "  MIN_LR=${MIN_LR}"
    echo "  WEIGHT_DECAY=${WEIGHT_DECAY}"
    echo "  WARMUP_EPOCHS=${WARMUP_EPOCHS}"
    echo "  TEMPERATURE=${TEMPERATURE}"
    echo "  SAMPLING=${SAMPLING}"
    echo "Loss configuration is defined by the A/B/C wrappers."

    run_experiment "A" "Full D Only" \
        "${SCRIPT_DIR}/train_full_d_only.sh" || exit $?
    run_experiment "B" "Full Prototype D" \
        "${SCRIPT_DIR}/train_full_prototype_d.sh" || exit $?
    run_experiment "C" "Missing Prototype D" \
        "${SCRIPT_DIR}/train_missing_prototype_d.sh" || exit $?

    echo
    echo "ALL EXPERIMENTS COMPLETED"
    echo "Completion time: $(date --iso-8601=seconds)"
}

if [[ "${1:-}" == "--worker" ]]; then
    if [[ "$#" -ne 2 ]]; then
        echo "Internal usage: $0 --worker MASTER_DIR" >&2
        exit 2
    fi
    run_worker "$2"
    exit 0
fi

if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "Missing required command: setsid" >&2
    exit 1
fi

export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export TEMPERATURE="${TEMPERATURE:-0.2}"
export SAMPLING="${SAMPLING:-cslpae}"

# Experiment-specific values must come from each wrapper, not the caller's shell.
unset RUN_PREFIX RUN_NAME OUTPUT_DIR
unset SCOPE MISSING_FILL OUTPUT_BASE COMPONENT_MODE COMPOSITION_MODE
unset SUB_CONTRA_WEIGHT TASK_CONTRA_WEIGHT SWAP_SUB_WEIGHT SWAP_TASK_WEIGHT
unset RECON_WEIGHT MISSING_MSE_WEIGHT

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_RUN_NAME="abc_four_loss_seed${SEED}_${TIMESTAMP}"
MASTER_OUTPUT_ROOT="${ABC_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/abc_four_loss}"
MASTER_DIR="${MASTER_OUTPUT_ROOT}/${MASTER_RUN_NAME}"
MASTER_LOG="${MASTER_DIR}/master.log"
MASTER_PID_FILE="${MASTER_DIR}/master.pid"

if [[ -e "${MASTER_DIR}" ]]; then
    echo "Refusing to overwrite existing master directory: ${MASTER_DIR}" >&2
    exit 1
fi
mkdir -p "${MASTER_DIR}"

nohup setsid bash "${BASH_SOURCE[0]}" --worker "${MASTER_DIR}" \
    >"${MASTER_LOG}" 2>&1 </dev/null &
MASTER_PID=$!
printf '%s\n' "${MASTER_PID}" >"${MASTER_PID_FILE}"

echo "ABC four-loss experiments started in background"
echo "Master PID: ${MASTER_PID}"
echo "Master log: ${MASTER_LOG}"
echo "Master output: ${MASTER_DIR}"
echo
echo "Experiment order:"
echo "A Full D Only"
echo "B Full Prototype D"
echo "C Missing Prototype D"
echo
echo "Follow: tail -f $(printf '%q' "${MASTER_LOG}")"
echo "Stop: kill ${MASTER_PID}"
