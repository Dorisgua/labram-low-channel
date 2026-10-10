#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
# 默认读取当前运行 Stage2 的最后一轮（epoch 19），与旧图的选轮口径一致。
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/stage2/checkpoint-19.pth}"
# 前台运行；Python 默认 max-samples=99999，即使用全部 Test 样本。
"${PYTHON}" -u "${SCRIPT_DIR}/plot_z_d_few_all.py" --checkpoint "${CHECKPOINT}" "$@" 2>&1 | tee -a "${SCRIPT_DIR}/plot_z_d_few_all.log"
