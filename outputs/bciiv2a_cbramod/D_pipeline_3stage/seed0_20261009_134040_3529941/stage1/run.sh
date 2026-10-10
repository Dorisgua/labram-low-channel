#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"

# 1. 样本入口：all=全部样本；random=随机抽取 SAMPLE_COUNT 个样本。
SAMPLE_MODE="${SAMPLE_MODE:-all}"
SAMPLE_COUNT="${SAMPLE_COUNT:-2000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

# 2. 数据集入口：test=测试集；train=训练集（沿用训练时的划分和归一化）。
SPLIT="${SPLIT:-test}"

# 3. 绘图入口：full=只画第一行；all=三行都画（z 和 d 各一组）。
# 第一行：输入22个真实导联，观察全部22个位置。
# 第二行：仍输入22个真实导联，只观察待补全的9个位置。
# 第三行：输入13个真实导联+9个Prototype，只观察这9个位置。
# 前两行检查选择位置的影响；后两行检查真实输入换为Prototype的影响。full/all
VIEWS="${VIEWS:-full}"

# 默认读取同目录的 Stage1 验证集最佳 checkpoint。
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/checkpoint-best.pth}"
# 可直接修改上面的默认值，也可通过环境变量或末尾 Python 参数覆盖。
# 例：SAMPLE_MODE=random SAMPLE_COUNT=2000 SPLIT=train VIEWS=full bash run.sh
# 默认输出按集合、样本模式和视图分目录；--output-dir 可指定其他目录。
"${PYTHON}" -u "${SCRIPT_DIR}/plot_z_d_few_all.py" \
    --checkpoint "${CHECKPOINT}" \
    --sample-mode "${SAMPLE_MODE}" --max-samples "${SAMPLE_COUNT}" \
    --sample-seed "${SAMPLE_SEED}" --split "${SPLIT}" --views "${VIEWS}" \
    "$@" 2>&1 | tee -a "${SCRIPT_DIR}/plot_z_d_few_all.log"
