#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LABRAM_ROOT="${LABRAM_BASE_ROOT:-${PROJECT_ROOT}}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/checkpoint-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation/diagnostics_observed_plus_d}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -u "${SCRIPT_DIR}/diagnose_c_observed_plus_d_probe.py" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
