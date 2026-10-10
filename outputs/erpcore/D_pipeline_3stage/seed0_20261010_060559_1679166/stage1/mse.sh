#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${STAGE_DIR}/../../../../.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"

# 只计算固定 Prototype 的测试集 missing MSE。
"${PYTHON}" -u tools/prototype_mse/evaluate_fixed_prototype_mse.py \
    --checkpoint "${STAGE_DIR}/checkpoint-best.pth" \
    --output "${STAGE_DIR}/prototype_mse_baseline/metrics_test.json" \
    --split test \
    --batch-size 128 \
    --num-workers 4 \
    "$@"
