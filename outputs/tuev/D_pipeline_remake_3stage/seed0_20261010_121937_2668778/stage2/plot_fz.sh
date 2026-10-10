#!/usr/bin/env bash
set -euo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${STAGE_DIR}/../../../../.." && pwd)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"

# TUEV 13→23 设置可选的缺失导联（大写）：F7、F8、T5、T6、A1、A2、FZ、PZ、T1、T2。
# 默认 FZ；可在下方修改 --channel，或运行 bash plot_fz.sh --channel PZ。
# 更换导联时同步修改 --output-dir，或通过 OUTPUT_DIR 指定新目录，避免覆盖原结果。
"${PYTHON}" -u tools/plot_channel_feature_comparison.py \
    --checkpoint "${CHECKPOINT:-${STAGE_DIR}/checkpoint-best.pth}" \
    --channel FZ --group-by class \
    --split test --patch-index 0 \
    --device "${DEVICE:-cpu}" \
    --output-dir "${OUTPUT_DIR:-${STAGE_DIR}/test_fz_first_trial_by_class}" \
    "$@"
