#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
: "${CHECKPOINT:?Please set CHECKPOINT}"
SPLIT="${SPLIT:-test}"
case "${SPLIT}" in val|test|both) ;; *) echo "SPLIT must be val, test, or both" >&2; exit 2 ;; esac
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint missing: ${CHECKPOINT}" >&2; exit 2; }
CHECKPOINT_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
OUTPUT="${OUTPUT:-${CHECKPOINT_DIR}/prototype_mse_baseline/metrics_${SPLIT}.json}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
printf 'Checkpoint: %s
Split: %s
Output: %s
' "${CHECKPOINT}" "${SPLIT}" "${OUTPUT}"
exec "${PYTHON}" -u "${REPO_DIR}/tools/prototype_mse/evaluate_fixed_prototype_mse.py" \
    --checkpoint "${CHECKPOINT}" --output "${OUTPUT}" --split "${SPLIT}" \
    --batch-size "${BATCH_SIZE:-128}" --num-workers "${NUM_WORKERS:-4}" "$@"
