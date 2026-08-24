#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${STAGE2_ROOT}/.." && pwd)"
LABRAM_BASE_ROOT="${LABRAM_BASE_ROOT:-${PROJECT_ROOT}}"
LABRAM_BASE_ROOT="$(cd "${LABRAM_BASE_ROOT}" && pwd)"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
export LABRAM_BASE_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_NAME="${EXP_NAME:-stage2_default}"
INPUT_MODE="${INPUT_MODE:-dynamic}"
SEED="${SEED:-0}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-${PROJECT_ROOT}/outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_RUN_DIR}/checkpoints/checkpoint-last.pth}"
STAGE1_CONFIG="${STAGE1_CONFIG:-${STAGE1_RUN_DIR}/config.json}"
LABRAM_CKPT="${LABRAM_CKPT:-${LABRAM_BASE_ROOT}/checkpoints/labram-base.pth}"
PROTOTYPE_CKPT="${PROTOTYPE_CKPT:-${LABRAM_BASE_ROOT}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/CSLP-AE/data_preparation/simple_data.pt}"
LAST_N_BLOCKS="${LAST_N_BLOCKS:-12}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-4}"
TRAIN_CNN="${TRAIN_CNN:-0}"
CNN_LR_MULT="${CNN_LR_MULT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stage2}"
BACKGROUND="${BACKGROUND:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_AUDIT="${RUN_AUDIT:-0}"
# RESUME="${RESUME:-}"

make_abs() {
    if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "${PROJECT_ROOT}" "$1"; fi
}
STAGE1_RUN_DIR="$(make_abs "${STAGE1_RUN_DIR}")"
STAGE1_CKPT="$(make_abs "${STAGE1_CKPT}")"
STAGE1_CONFIG="$(make_abs "${STAGE1_CONFIG}")"
DATA_PATH="$(make_abs "${DATA_PATH}")"
OUTPUT_ROOT="$(make_abs "${OUTPUT_ROOT}")"
# if [[ -n "${RESUME}" ]]; then RESUME="$(make_abs "${RESUME}")"; fi

case "${INPUT_MODE}" in full|observed_only|prototype|dynamic) ;; *) echo "INPUT_MODE must be full, observed_only, prototype, or dynamic" >&2; exit 2 ;; esac
if [[ "${LAST_N_BLOCKS}" -lt 0 ]]; then echo "LAST_N_BLOCKS must be non-negative" >&2; exit 2; fi
if [[ "${TRAIN_CNN}" != "0" && "${TRAIN_CNN}" != "1" ]]; then echo "TRAIN_CNN must be 0 or 1" >&2; exit 2; fi
if [[ "${INPUT_MODE}" == "dynamic" && "${TRAIN_CNN}" == "1" ]]; then echo "dynamic mode keeps the Stage1 TemporalConv frozen; TRAIN_CNN must be 0" >&2; exit 2; fi
if [[ ! -x "${PYTHON}" ]]; then echo "Missing Python: ${PYTHON}" >&2; exit 1; fi
if [[ ! -f "${LABRAM_CKPT}" ]]; then echo "Missing LABRAM_CKPT: ${LABRAM_CKPT}" >&2; exit 1; fi
if [[ "${INPUT_MODE}" == "prototype" || "${INPUT_MODE}" == "dynamic" ]] && [[ ! -f "${PROTOTYPE_CKPT}" ]]; then echo "Missing PROTOTYPE_CKPT: ${PROTOTYPE_CKPT}" >&2; exit 1; fi
if [[ "${INPUT_MODE}" == "dynamic" ]]; then
    [[ -f "${STAGE1_CKPT}" ]] || { echo "Missing STAGE1_CKPT: ${STAGE1_CKPT}" >&2; exit 1; }
    [[ -f "${STAGE1_CONFIG}" ]] || { echo "Missing STAGE1_CONFIG: ${STAGE1_CONFIG}" >&2; exit 1; }
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
CMD=("${PYTHON}" -u -m clean_disentangle.stage2.train_stage2
    --exp-name "${EXP_NAME}" --input-mode "${INPUT_MODE}"
    --stage1-checkpoint "${STAGE1_CKPT}" --stage1-config "${STAGE1_CONFIG}"
    --labram-checkpoint "${LABRAM_CKPT}" --prototype-checkpoint "${PROTOTYPE_CKPT}"
    --data-path "${DATA_PATH}"
    --output-dir "${OUTPUT_DIR}" --device "${DEVICE}" --seed "${SEED}"
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"
    --lr "${LR}" --cnn-lr-mult "${CNN_LR_MULT}" --weight-decay "${WEIGHT_DECAY}" --warmup-epochs "${WARMUP_EPOCHS}"
    --last-n-blocks "${LAST_N_BLOCKS}")
[[ "${TRAIN_CNN}" == "1" ]] && CMD+=(--train-cnn)
[[ "${DRY_RUN}" == "1" ]] && CMD+=(--dry-run)
[[ "${RUN_AUDIT}" == "1" ]] && CMD+=(--audit-only)
# [[ -n "${RESUME}" ]] && CMD+=(--resume "${RESUME}")

echo "Resolved Stage2 launcher: EXP_NAME=${EXP_NAME} INPUT_MODE=${INPUT_MODE} LAST_N_BLOCKS=${LAST_N_BLOCKS} SEED=${SEED} OUTPUT_DIR=${OUTPUT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
    "${CMD[@]}"
    exit $?
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run.log"
if [[ "${BACKGROUND}" == "0" ]]; then
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    exit "${PIPESTATUS[0]}"
fi
nohup setsid "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" >"${OUTPUT_DIR}/train.pid"
echo "Stage2 ${INPUT_MODE} started in background"
echo "Run: ${RUN_NAME}"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
