#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
R0_RUN_DIR="${PROJECT_ROOT}/outputs/full_d_only/full_d_only_seed0_20260818_131233"
R0_CONFIG="${R0_RUN_DIR}/config.json"
R0_CHECKPOINT="${R0_RUN_DIR}/checkpoints/checkpoint-last.pth"

check_r0_reference() {
    "${PYTHON}" -c '
import json, math, sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
expected = {
    "dataset": "erpcore",
    "seed": 0,
    "scope": "full",
    "missing_fill": "not_applicable",
    "output_base": "none",
    "component_mode": "identity",
    "composition_mode": "sum",
    "sub_contra_weight": 1.0,
    "task_contra_weight": 1.0,
    "swap_sub_weight": 1.0,
    "swap_task_weight": 1.0,
    "recon_weight": 0.0,
    "missing_mse_weight": 0.0,
    "sampler": "cslpae",
    "temperature": 0.2,
    "batch_size": 64,
    "epochs": 50,
    "optimizer": "adamw",
    "opt_eps": 1e-8,
    "lr": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "schedule": "cosine",
    "trainable_parameter_count": 1449000,
    "trainable_scope": "stable_core_only",
    "patch_embedding": "frozen_eval",
}
differences = []
for key, value in expected.items():
    actual = config.get(key)
    if isinstance(value, float):
        matches = isinstance(actual, (int, float)) and math.isclose(
            float(actual), value, rel_tol=0.0, abs_tol=1e-12
        )
    else:
        matches = actual == value
    if not matches:
        differences.append(f"{key}: expected={value!r}, actual={actual!r}")
if differences:
    raise SystemExit("R0 config mismatch; refusing to launch sweep:\n  " + "\n  ".join(differences))
print(f"R0 reference verified: {path.parent}")
' "${R0_CONFIG}"
}

run_worker() {
    local master_dir="$1"
    local sweep_timestamp="$2"
    local active_pid=""

    stop_active_experiment() {
        if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
            kill "${active_pid}" 2>/dev/null || true
            wait "${active_pid}" 2>/dev/null || true
        fi
        echo "Sweep master stopped at $(date --iso-8601=seconds)"
        exit 143
    }
    trap stop_active_experiment TERM INT

    run_probe() {
        local checkpoint="$1"
        local run_dir="$2"
        if [[ "${RUN_PROBE}" != "1" ]]; then
            echo "Probe skipped because RUN_PROBE=${RUN_PROBE}"
            return 0
        fi
        echo "Probe start time: $(date --iso-8601=seconds)"
        "${PYTHON}" -u -m clean_disentangle.evaluation.probe_latents \
            --checkpoint "${checkpoint}" \
            --output-dir "${run_dir}/evaluation/probe" \
            --device "${PROBE_DEVICE}" \
            --batch-size "${PROBE_BATCH_SIZE}" \
            --num-workers "${PROBE_NUM_WORKERS}" \
            --cv-seed 42 \
            --xgb-seed 42 \
            --undersample-seed 42
        echo "Probe finish time: $(date --iso-8601=seconds)"
    }

    run_experiment() {
        local run_id="$1"
        local cnn_state="$2"
        local task_weight="$3"
        local swap_sub_weight="$4"
        local swap_task_weight="$5"
        local unfreeze_cnn="$6"
        local run_name
        local run_dir
        local checkpoint
        local exit_code

        run_name="${run_id}_full_d_${cnn_state}_w1_${task_weight}_${swap_sub_weight}_${swap_task_weight}_seed0_${sweep_timestamp}"
        run_dir="${SWEEP_OUTPUT_ROOT}/${run_name}"
        checkpoint="${run_dir}/checkpoints/checkpoint-last.pth"

        echo
        echo "===== START ${run_id^^}: ${run_name} ====="
        echo "Run directory: ${run_dir}"
        echo "Start time: $(date --iso-8601=seconds)"
        echo "Resolved sweep variables: CNN=${cnn_state}, SubC=1, TaskC=${task_weight}, SubSwap=${swap_sub_weight}, TaskSwap=${swap_task_weight}"

        BACKGROUND=0 \
        RUN_PREFIX="${run_id}_full_d" \
        RUN_NAME="${run_name}" \
        OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT}" \
        SCOPE=full \
        MISSING_FILL=not_applicable \
        OUTPUT_BASE=none \
        COMPONENT_MODE=identity \
        COMPOSITION_MODE=sum \
        SUB_CONTRA_WEIGHT=1 \
        TASK_CONTRA_WEIGHT="${task_weight}" \
        SWAP_SUB_WEIGHT="${swap_sub_weight}" \
        SWAP_TASK_WEIGHT="${swap_task_weight}" \
        RECON_WEIGHT=0 \
        MISSING_MSE_WEIGHT=0 \
        UNFREEZE_CNN="${unfreeze_cnn}" \
        CNN_LR_MULT=0.1 \
        bash "${SCRIPT_DIR}/train_full_d_only.sh" &
        active_pid=$!
        set +e
        wait "${active_pid}"
        exit_code=$?
        set -e
        active_pid=""

        echo "Training finish time: $(date --iso-8601=seconds)"
        echo "Training exit code: ${exit_code}"
        if [[ "${exit_code}" -ne 0 ]]; then
            echo "FAILED AT EXPERIMENT ${run_id^^}"
            return "${exit_code}"
        fi
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN verified launcher for ${run_id^^}; checkpoint/probe skipped"
            return 0
        fi
        if [[ ! -f "${checkpoint}" ]]; then
            echo "Missing completed checkpoint: ${checkpoint}" >&2
            echo "FAILED AT EXPERIMENT ${run_id^^}" >&2
            return 1
        fi
        run_probe "${checkpoint}" "${run_dir}" || {
            echo "FAILED DURING PROBE FOR ${run_id^^}" >&2
            return 1
        }
        echo "===== FINISH ${run_id^^}: ${run_name} ====="
    }

    echo "CSLP reproduction sweep started at $(date --iso-8601=seconds)"
    echo "Master directory: ${master_dir}"
    check_r0_reference
    echo "R0 checkpoint retained without retraining: ${R0_CHECKPOINT}"
    echo "Locked common configuration:"
    echo "  seed=0 epochs=50 batch_size=64"
    echo "  lr=5e-4 cnn_lr_mult=0.1 min_lr=1e-6"
    echo "  weight_decay=0.05 warmup_epochs=5 temperature=0.2"
    echo "  scope=full output_base=none component=identity composition=sum"
    echo "  sampler=cslpae recon_weight=0 missing_mse_weight=0"
    echo "  probe_after_train=${RUN_PROBE}"

    run_experiment r1 unfreeze 1 1 1 1 || exit $?
    run_experiment r2 frozen 5 1 1 0 || exit $?
    run_experiment r3 frozen 10 1 1 0 || exit $?
    run_experiment r4 frozen 1 20 20 0 || exit $?
    run_experiment r5 unfreeze 5 20 20 1 || exit $?

    echo
    echo "ALL R1-R5 EXPERIMENTS AND PROBES COMPLETED"
    echo "Completion time: $(date --iso-8601=seconds)"
}

if [[ "${1:-}" == "--worker" ]]; then
    if [[ "$#" -ne 3 ]]; then
        echo "Internal usage: $0 --worker MASTER_DIR SWEEP_TIMESTAMP" >&2
        exit 2
    fi
    run_worker "$2" "$3"
    exit 0
fi

if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing executable Python: ${PYTHON}" >&2
    exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "Missing required command: setsid" >&2
    exit 1
fi
if [[ ! -f "${R0_CONFIG}" || ! -f "${R0_CHECKPOINT}" ]]; then
    echo "R0 reference config/checkpoint is missing under: ${R0_RUN_DIR}" >&2
    exit 1
fi

# This sweep is intentionally locked to the current A reference.
export SEED=0
export EPOCHS=50
export BATCH_SIZE=64
export LR=5e-4
export MIN_LR=1e-6
export WEIGHT_DECAY=0.05
export WARMUP_EPOCHS=5
export TEMPERATURE=0.2
export SAMPLING=cslpae
export RUN_PROBE="${RUN_PROBE:-1}"
export PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
export PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
export PROBE_NUM_WORKERS="${PROBE_NUM_WORKERS:-4}"
export DRY_RUN="${DRY_RUN:-0}"
export SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/cslp_repro_sweep}"

for existing_pid_file in "${SWEEP_OUTPUT_ROOT}"/masters/*/master.pid; do
    [[ -f "${existing_pid_file}" ]] || continue
    existing_pid="$(<"${existing_pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Refusing to start a duplicate sweep; active master PID ${existing_pid}" >&2
        echo "PID file: ${existing_pid_file}" >&2
        exit 1
    fi
done

unset RUN_PREFIX RUN_NAME OUTPUT_DIR
unset SCOPE MISSING_FILL OUTPUT_BASE COMPONENT_MODE COMPOSITION_MODE
unset SUB_CONTRA_WEIGHT TASK_CONTRA_WEIGHT SWAP_SUB_WEIGHT SWAP_TASK_WEIGHT
unset RECON_WEIGHT MISSING_MSE_WEIGHT UNFREEZE_CNN CNN_LR_MULT BACKGROUND

check_r0_reference

SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_DIR="${SWEEP_OUTPUT_ROOT}/masters/cslp_repro_sweep_seed0_${SWEEP_TIMESTAMP}"
MASTER_LOG="${MASTER_DIR}/master.log"
MASTER_PID_FILE="${MASTER_DIR}/master.pid"

if [[ -e "${MASTER_DIR}" ]]; then
    echo "Refusing to overwrite existing master directory: ${MASTER_DIR}" >&2
    exit 1
fi
mkdir -p "${MASTER_DIR}"

nohup setsid bash "${BASH_SOURCE[0]}" --worker "${MASTER_DIR}" "${SWEEP_TIMESTAMP}" \
    >"${MASTER_LOG}" 2>&1 </dev/null &
MASTER_PID=$!
printf '%s\n' "${MASTER_PID}" >"${MASTER_PID_FILE}"

sleep 2
if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
    set +e
    wait "${MASTER_PID}"
    EXIT_CODE=$?
    set -e
    if [[ "${EXIT_CODE}" == "0" && "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN completed successfully"
        echo "Master log: ${MASTER_LOG}"
        exit 0
    fi
    echo "Sweep master exited during startup with code ${EXIT_CODE}" >&2
    echo "Master log: ${MASTER_LOG}" >&2
    tail -n 60 "${MASTER_LOG}" >&2
    exit "${EXIT_CODE}"
fi

echo "CSLP reproduction sweep R1-R5 started in background"
echo "R0: verified existing reference; not retrained"
echo "Master PID: ${MASTER_PID}"
echo "Master log: ${MASTER_LOG}"
echo "Master output: ${MASTER_DIR}"
echo "Experiment output root: ${SWEEP_OUTPUT_ROOT}"
echo
echo "Experiment order: R1 -> R2 -> R3 -> R4 -> R5"
echo "Follow: tail -f $(printf '%q' "${MASTER_LOG}")"
echo "Stop: kill ${MASTER_PID}"
