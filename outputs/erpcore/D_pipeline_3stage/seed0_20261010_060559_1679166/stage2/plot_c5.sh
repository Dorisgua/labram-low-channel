#!/usr/bin/env bash
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${STAGE_DIR}/../../../../.." && pwd)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"

"${PYTHON}" -u tools/plot_channel_feature_comparison.py \
    --checkpoint "${CHECKPOINT:-${STAGE_DIR}/checkpoint-best.pth}" \
    --channel C5 --group-by subject \
    --split test --patch-index 0 \
    --device "${DEVICE:-cpu}" \
    --output-dir "${OUTPUT_DIR:-${STAGE_DIR}/test_c5_first_trial_by_subject}" \
    "$@"
