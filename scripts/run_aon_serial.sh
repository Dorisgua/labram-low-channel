#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
if [[ $# -eq 0 ]]; then set -- shu seedv tuev seed erp_core; fi
read -r -a modes <<< "${AON_MODES:-A N O}"
[[ ${#modes[@]} -gt 0 ]] || { echo 'AON_MODES is empty' >&2; exit 2; }
for dataset in "$@"; do
    case "$dataset" in shu|seedv|tuev|seed|erp_core) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
    for mode in "${modes[@]}"; do
        case "$mode" in A|N|O) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
        [[ -f "${SCRIPT_DIR}/${dataset}/${mode}/freeze_cnn.sh" ]] || exit 2
    done
done
export SEED="${SEED:-0}"
export AON_RUN_ID="${AON_RUN_ID:-seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$}"
export AON_LOG_ROOT="${AON_LOG_ROOT:-${REPO_DIR}/outputs/aon_serial/${AON_RUN_ID}}"

run_one() (
    dataset="$1"
    mode="$2"
    output_name="$dataset"
    [[ "$dataset" != erp_core ]] || output_name=erpcore
    unset OUTPUT_DIR OUTPUT_ROOT OUTPUT_SCRIPT_NAME TB_LOG_DIR TERMINAL_LOG_DIR
    unset DATA_PATH CHANNEL_PROTOTYPE_PATH FINETUNE RESUME STAGE1_CHECKPOINT
    unset CHANNEL_SUBSET COMPLETION_SCOPE POOLING_SCOPE TRAIN_ENTRYPOINT MODEL
    export MODEL=labram_base_patch200_200 TRAIN_ENTRYPOINT=run_class_finetuning.py
    export RUN_FOREGROUND=1 NO_AUTO_RESUME=1 EVAL_ONLY=0 MASTER_PORT=auto
    export OUTPUT_DIR="${REPO_DIR}/outputs/${output_name}/aon/${AON_RUN_ID}/${mode}"
    # Optional dataset-specific data locations; otherwise use the existing base.
    path_key="${dataset^^}_DATA_PATH"
    if [[ -n "${!path_key:-}" ]]; then export DATA_PATH="${!path_key}"; fi
    echo "[$(date -Is)] ${dataset}/${mode}: ${OUTPUT_DIR}"
    bash "${SCRIPT_DIR}/${dataset}/${mode}/freeze_cnn.sh"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    for dataset in "$@"; do
        for mode in "${modes[@]}"; do run_one "$dataset" "$mode"; done
    done
    exit 0
fi
if [[ "${AON_SERIAL_CHILD:-0}" != 1 ]]; then
    mkdir -p "${AON_LOG_ROOT}"
    nohup setsid env AON_SERIAL_CHILD=1 bash "${SCRIPT_DIR}/run_aon_serial.sh" "$@" \
        > "${AON_LOG_ROOT}/serial.log" 2>&1 < /dev/null &
    serial_pid=$!
    echo "$serial_pid" > "${AON_LOG_ROOT}/launcher.pid"
    echo "Started A/N/O queue; PID: ${serial_pid}"
    echo "Datasets: $*; modes: ${modes[*]}"
    echo "Log: ${AON_LOG_ROOT}/serial.log"
    exit 0
fi
mkdir -p "${AON_LOG_ROOT}"
printf 'dataset\tmode\tstatus\tfinished_at\n' > "${AON_LOG_ROOT}/status.tsv"
for dataset in "$@"; do
    for mode in "${modes[@]}"; do
        set +e
        (set -e; run_one "$dataset" "$mode")
        result=$?
        set -e
        if [[ "$result" -ne 0 ]]; then
            printf '%s\t%s\tfailed:%s\t%s\n' "$dataset" "$mode" "$result" "$(date -Is)" >> "${AON_LOG_ROOT}/status.tsv"
            exit "$result"
        fi
        printf '%s\t%s\tcompleted\t%s\n' "$dataset" "$mode" "$(date -Is)" >> "${AON_LOG_ROOT}/status.tsv"
    done
done
echo 'All A/N/O experiments completed.'
