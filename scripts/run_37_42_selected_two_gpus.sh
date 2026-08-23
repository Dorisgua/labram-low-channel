#!/usr/bin/env bash
set -euo pipefail

# Run the selected FACED and EEGMAT experiments with two GPUs.
# It starts two FACED jobs immediately, then a small monitor starts EEGMAT19
# after both FACED jobs exit successfully.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG_DIR="${LAUNCHER_LOG_DIR:-outputs/launcher_logs}"
mkdir -p "${LAUNCHER_LOG_DIR}"

FACED_FULL_SCRIPT="scripts/faced/42Oada.finetune_faced32_labrambase_mlp_full_finetune.sh"
FACED_FREEZE_SCRIPT="scripts/faced/42Oada.finetune_faced32_labrambase_mlp_freeze_cnn.sh"
EEGMAT_FREEZE_SCRIPT="scripts/eegmat/37Oada.finetune_eegmat19_labrambase_freeze_cnn.sh"

for script in "${FACED_FULL_SCRIPT}" "${FACED_FREEZE_SCRIPT}" "${EEGMAT_FREEZE_SCRIPT}"; do
    if [[ ! -f "${script}" ]]; then
        echo "Missing script: ${script}" >&2
        exit 1
    fi
done

start_job() {
    local name="$1"
    local gpu="$2"
    local port="$3"
    local script="$4"
    local log="${LAUNCHER_LOG_DIR}/${RUN_ID}_${name}.log"
    local status="${LAUNCHER_LOG_DIR}/${RUN_ID}_${name}.status"
    local pid_file="${LAUNCHER_LOG_DIR}/${RUN_ID}_${name}.pid"

    rm -f "${status}" "${pid_file}"
    setsid nohup bash -c '
        set +e
        export GPU_IDS="$1"
        export CUDA_VISIBLE_DEVICES="$1"
        export MASTER_PORT="$2"
        export RUN_FOREGROUND=1
        bash "$3"
        echo "$?" > "$4"
    ' _ "${gpu}" "${port}" "${script}" "${status}" > "${log}" 2>&1 &
    LAST_PID="$!"
    echo "${LAST_PID}" > "${pid_file}"
}

echo "Run id: ${RUN_ID}"
echo "Batch 1: start FACED full fine-tune on GPU ${GPU0}"
start_job faced_full "${GPU0}" "${FACED_FULL_MASTER_PORT:-29524}" "${FACED_FULL_SCRIPT}"
pid_faced_full="${LAST_PID}"
echo "  PID ${pid_faced_full}; log ${LAUNCHER_LOG_DIR}/${RUN_ID}_faced_full.log"

echo "Batch 1: start FACED freeze-CNN on GPU ${GPU1}"
start_job faced_freeze "${GPU1}" "${FACED_FREEZE_MASTER_PORT:-29525}" "${FACED_FREEZE_SCRIPT}"
pid_faced_freeze="${LAST_PID}"
echo "  PID ${pid_faced_freeze}; log ${LAUNCHER_LOG_DIR}/${RUN_ID}_faced_freeze.log"

monitor_log="${LAUNCHER_LOG_DIR}/${RUN_ID}_monitor.log"
setsid nohup bash -c '
    set +e
    run_id="$1"
    launcher_log_dir="$2"
    pid_faced_full="$3"
    pid_faced_freeze="$4"
    gpu0="$5"
    eegmat_port="$6"
    eegmat_script="$7"
    monitor_interval="${MONITOR_INTERVAL_SECONDS:-60}"

    echo "Monitor started for run ${run_id}"
    echo "Waiting for FACED PIDs: ${pid_faced_full}, ${pid_faced_freeze}"
    while kill -0 "${pid_faced_full}" 2>/dev/null || kill -0 "${pid_faced_freeze}" 2>/dev/null; do
        sleep "${monitor_interval}"
    done

    full_status="$(cat "${launcher_log_dir}/${run_id}_faced_full.status" 2>/dev/null || echo 127)"
    freeze_status="$(cat "${launcher_log_dir}/${run_id}_faced_freeze.status" 2>/dev/null || echo 127)"
    echo "FACED full exit code: ${full_status}"
    echo "FACED freeze exit code: ${freeze_status}"
    if [[ "${full_status}" != "0" || "${freeze_status}" != "0" ]]; then
        echo "At least one FACED job failed; EEGMAT19 will not start"
        exit 1
    fi

    echo "Batch 2: start EEGMAT19 AdaBrain-style freeze-CNN on GPU ${gpu0}"
    eegmat_status="${launcher_log_dir}/${run_id}_eegmat19_freeze.status"
    eegmat_pid_file="${launcher_log_dir}/${run_id}_eegmat19_freeze.pid"
    eegmat_log="${launcher_log_dir}/${run_id}_eegmat19_freeze.log"
    rm -f "${eegmat_status}" "${eegmat_pid_file}"
    setsid nohup bash -c '"'"'
        set +e
        export GPU_IDS="$1"
        export CUDA_VISIBLE_DEVICES="$1"
        export MASTER_PORT="$2"
        export RUN_FOREGROUND=1
        bash "$3"
        echo "$?" > "$4"
    '"'"' _ "${gpu0}" "${eegmat_port}" "${eegmat_script}" "${eegmat_status}" > "${eegmat_log}" 2>&1 &
    eegmat_pid="$!"
    echo "${eegmat_pid}" > "${eegmat_pid_file}"
    echo "  PID ${eegmat_pid}; log ${eegmat_log}"
' _ "${RUN_ID}" "${LAUNCHER_LOG_DIR}" "${pid_faced_full}" "${pid_faced_freeze}" "${GPU0}" "${EEGMAT_FREEZE_MASTER_PORT:-29526}" "${EEGMAT_FREEZE_SCRIPT}" > "${monitor_log}" 2>&1 &
monitor_pid="$!"

echo "Monitor PID ${monitor_pid}; log ${monitor_log}"
