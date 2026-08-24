#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABRAM_BASE_ROOT="${LABRAM_BASE_ROOT:-${PROJECT_ROOT}}"
LABRAM_BASE_ROOT="$(cd "${LABRAM_BASE_ROOT}" && pwd)"
export LABRAM_BASE_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
RUN_PREFIX="${RUN_PREFIX:?wrapper must set RUN_PREFIX}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
UNFREEZE_CNN="${UNFREEZE_CNN:-0}"
CNN_LR_MULT="${CNN_LR_MULT:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
OPT_EPS="${OPT_EPS:-1e-8}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULE="${SCHEDULE:-cosine}"
TEMPERATURE="${TEMPERATURE:-0.2}"

SCOPE="${SCOPE:?wrapper must set SCOPE}"
MISSING_FILL="${MISSING_FILL:?wrapper must set MISSING_FILL}"
OUTPUT_BASE="${OUTPUT_BASE:?wrapper must set OUTPUT_BASE}"
COMPONENT_MODE="${COMPONENT_MODE:-identity}"
COMPOSITION_MODE="${COMPOSITION_MODE:-sum}"
SUB_CONTRA_WEIGHT="${SUB_CONTRA_WEIGHT:-0}"
TASK_CONTRA_WEIGHT="${TASK_CONTRA_WEIGHT:-0}"
SWAP_SUB_WEIGHT="${SWAP_SUB_WEIGHT:-0}"
SWAP_TASK_WEIGHT="${SWAP_TASK_WEIGHT:-0}"
RECON_WEIGHT="${RECON_WEIGHT:-0}"
MISSING_MSE_WEIGHT="${MISSING_MSE_WEIGHT:-0}"
SAMPLING="${SAMPLING:-cslpae}"

DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/CSLP-AE/data_preparation/simple_data.pt}"
CNN_CHECKPOINT="${CNN_CHECKPOINT:-${LABRAM_BASE_ROOT}/checkpoints/labram-base.pth}"
PROTOTYPE_CHECKPOINT="${PROTOTYPE_CHECKPOINT:-${LABRAM_BASE_ROOT}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
NORM_METHOD="${NORM_METHOD:-z_score}"
INPUT_SCALE="${INPUT_SCALE:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEM="${PIN_MEM:-1}"
DEVICE="${DEVICE:-cuda}"
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DRY_RUN="${DRY_RUN:-0}"
BACKGROUND="${BACKGROUND:-1}"
STARTUP_CHECK_SECONDS="${STARTUP_CHECK_SECONDS:-2}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_seed${SEED}_${TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_PREFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_FILE="${OUTPUT_DIR}/run.log"
PID_FILE="${OUTPUT_DIR}/train.pid"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing executable Python: ${PYTHON}" >&2
    exit 1
fi
if [[ "${BACKGROUND}" != "0" && "${BACKGROUND}" != "1" ]]; then
    echo "BACKGROUND must be 0 or 1, got: ${BACKGROUND}" >&2
    exit 1
fi
if [[ "${UNFREEZE_CNN}" != "0" && "${UNFREEZE_CNN}" != "1" ]]; then
    echo "UNFREEZE_CNN must be 0 or 1, got: ${UNFREEZE_CNN}" >&2
    exit 1
fi
if [[ "${BACKGROUND}" == "1" ]] && ! command -v setsid >/dev/null 2>&1; then
    echo "Missing required command: setsid" >&2
    exit 1
fi
if [[ ! -f "${CNN_CHECKPOINT}" ]]; then
    echo "Missing CNN checkpoint: ${CNN_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${DATA_PATH}" && ! -f "${DATA_PATH}/simple_data.pt" ]]; then
    echo "Missing ERP-Core data: ${DATA_PATH}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to overwrite existing run directory: ${OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/checkpoints"

CMD=(
    "${PYTHON}"
    -u
    -m clean_disentangle.run
    --mode train
    --run-name "${RUN_NAME}"
    --output-dir "${OUTPUT_DIR}"
    --legacy-root "${LABRAM_BASE_ROOT}"
    --cnn-checkpoint "${CNN_CHECKPOINT}"
    --prototype-checkpoint "${PROTOTYPE_CHECKPOINT}"
    --dataset erpcore
    --data-path "${DATA_PATH}"
    --sampling-rate "${SAMPLING_RATE}"
    --norm-method "${NORM_METHOD}"
    --input-scale "${INPUT_SCALE}"
    --num-workers "${NUM_WORKERS}"
    --scope "${SCOPE}"
    --missing-fill "${MISSING_FILL}"
    --output-base "${OUTPUT_BASE}"
    --component-mode "${COMPONENT_MODE}"
    --composition-mode "${COMPOSITION_MODE}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --optimizer "${OPTIMIZER}"
    --opt-eps "${OPT_EPS}"
    --lr "${LR}"
    --cnn-lr-mult "${CNN_LR_MULT}"
    --min-lr "${MIN_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --schedule "${SCHEDULE}"
    --temperature "${TEMPERATURE}"
    --sampling "${SAMPLING}"
    --sub-contra-weight "${SUB_CONTRA_WEIGHT}"
    --task-contra-weight "${TASK_CONTRA_WEIGHT}"
    --swap-sub-weight "${SWAP_SUB_WEIGHT}"
    --swap-task-weight "${SWAP_TASK_WEIGHT}"
    --recon-weight "${RECON_WEIGHT}"
    --missing-mse-weight "${MISSING_MSE_WEIGHT}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --log-interval "${LOG_INTERVAL}"
    --seed "${SEED}"
    --device "${DEVICE}"
)
if [[ "${UNFREEZE_CNN}" == "1" ]]; then
    CMD+=(--unfreeze-cnn)
else
    CMD+=(--freeze-cnn)
fi
if [[ "${PIN_MEM}" == "1" ]]; then
    CMD+=(--pin-mem)
else
    CMD+=(--no-pin-mem)
fi

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q' \
        "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    if [[ "${BACKGROUND}" == "1" ]]; then
        printf ' nohup setsid'
    fi
    printf ' %q' "${CMD[@]}"
    printf '\n'
}

echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Actual command:"
print_command

{
    echo "Run name: ${RUN_NAME}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Command:"
    print_command
    echo
} >"${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: command was not started"
    exit 0
fi

if [[ "${BACKGROUND}" == "0" ]]; then
    printf '%s\n' "$$" >"${PID_FILE}"
    echo "Running training in blocking foreground mode"
    echo "PID: $$"
    exec "${CMD[@]}" >>"${LOG_FILE}" 2>&1 </dev/null
fi

nohup setsid "${CMD[@]}" >>"${LOG_FILE}" 2>&1 </dev/null &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"

sleep "${STARTUP_CHECK_SECONDS}"
if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "Started training in background"
    echo "Run: ${RUN_NAME}"
    echo "PID: ${TRAIN_PID}"
    echo "Log: ${LOG_FILE}"
    echo "Follow: tail -f $(printf '%q' "${LOG_FILE}")"
    echo "Stop: kill ${TRAIN_PID}"
    exit 0
fi

set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e
if [[ "${EXIT_CODE}" == "0" ]]; then
    echo "Training command completed during the startup check"
    echo "Run: ${RUN_NAME}"
    echo "Log: ${LOG_FILE}"
    exit 0
fi

echo "Training exited during startup with code ${EXIT_CODE}" >&2
echo "Log: ${LOG_FILE}" >&2
tail -n 40 "${LOG_FILE}" >&2
exit "${EXIT_CODE}"
