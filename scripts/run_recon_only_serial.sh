#!/usr/bin/env bash
set -euo pipefail

# Run existing two-stage pipelines sequentially with disentanglement losses off.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -eq 0 ]]; then
    set -- shu seedv tuev seed erp_core
fi
for dataset in "$@"; do
    case "${dataset}" in
        shu|seedv|tuev|seed|erp_core) ;;
        *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
    esac
    if [[ ! -f "${SCRIPT_DIR}/${dataset}/D/stage1_then_stage2.sh" ]]; then
        echo "Missing pipeline for ${dataset}" >&2
        exit 2
    fi
done

export SEED="${SEED:-0}"
export MISSING_WEIGHT="${MISSING_WEIGHT:-500}"
if ! awk -v value="${MISSING_WEIGHT}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }'; then
    echo "MISSING_WEIGHT must be a positive number for recon-only." >&2
    exit 2
fi
export RECON_RUN_ID="${RECON_RUN_ID:-seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$}"
export RECON_LOG_ROOT="${RECON_LOG_ROOT:-${REPO_DIR}/outputs/recon_only_serial/${RECON_RUN_ID}}"

run_dataset() (
    local dataset="$1"
    local output_name="${dataset}"
    [[ "${dataset}" != "erp_core" ]] || output_name="erpcore"
    # Prevent previous experiments' paths from overriding the new handoff.
    unset OUTPUT_DIR OUTPUT_ROOT OUTPUT_SCRIPT_NAME TB_LOG_DIR TERMINAL_LOG_DIR
    unset DATA_PATH CHANNEL_PROTOTYPE_PATH FINETUNE RESUME STAGE1_CHECKPOINT
    unset CHANNEL_SUBSET COMPLETION_SCOPE TRAIN_ENTRYPOINT MODEL
    export RUN_ROOT="${REPO_DIR}/outputs/${output_name}/recon_only/${RECON_RUN_ID}"
    export STAGE1_OUTPUT_DIR="${RUN_ROOT}/stage1"
    export STAGE2_OUTPUT_DIR="${RUN_ROOT}/stage2"
    export RUN_BACKGROUND=0 RUN_FOREGROUND=1 PIPELINE_CHILD=0
    export REG_WEIGHT=0
    export SUBJECT_SUMMARY_CONTRA_WEIGHT=0 TASK_SUMMARY_CONTRA_WEIGHT=0
    export SUBJECT_CORRECTION_CONTRA_WEIGHT=0 TASK_CORRECTION_CONTRA_WEIGHT=0
    export PERMUTE_SUB_WEIGHT=0 PERMUTE_TASK_WEIGHT=0
    echo "[$(date -Is)] Starting ${dataset}: ${RUN_ROOT}"
    bash "${SCRIPT_DIR}/${dataset}/D/stage1_then_stage2.sh"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for dataset in "$@"; do run_dataset "${dataset}"; done
    echo "Dry run completed (existing pipelines preview Stage1 and report Stage2 handoff)."
    exit 0
fi

if [[ "${RECON_SERIAL_CHILD:-0}" != "1" ]]; then
    mkdir -p "${RECON_LOG_ROOT}"
    nohup setsid env RECON_SERIAL_CHILD=1 bash "${BASH_SOURCE[0]}" "$@" \
        > "${RECON_LOG_ROOT}/serial.log" 2>&1 < /dev/null &
    serial_pid=$!
    echo "${serial_pid}" > "${RECON_LOG_ROOT}/launcher.pid"
    echo "Started serial recon-only queue; PID: ${serial_pid}"
    echo "Order: $*"
    echo "Log: ${RECON_LOG_ROOT}/serial.log"
    exit 0
fi

mkdir -p "${RECON_LOG_ROOT}"
printf 'dataset\tstatus\tfinished_at\n' > "${RECON_LOG_ROOT}/status.tsv"
for dataset in "$@"; do
    # Keep the pipeline outside an if-condition so its errexit remains active.
    set +e
    (set -e; run_dataset "${dataset}")
    result=$?
    set -e
    if [[ "${result}" -ne 0 ]]; then
        printf '%s\tfailed:%s\t%s\n' "${dataset}" "${result}" "$(date -Is)" >> "${RECON_LOG_ROOT}/status.tsv"
        echo "Stopped: ${dataset} failed with exit code ${result}."
        exit "${result}"
    fi
    printf '%s\tcompleted\t%s\n' "${dataset}" "$(date -Is)" >> "${RECON_LOG_ROOT}/status.tsv"
done
echo "All recon-only pipelines completed."
