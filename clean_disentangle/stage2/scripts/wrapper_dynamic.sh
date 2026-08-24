#!/usr/bin/env bash
set -euo pipefail
export EXP_NAME="${EXP_NAME:-dynamic_stage1C_last12_lr5e4_seed0}"
export INPUT_MODE="${INPUT_MODE:-dynamic}"
export STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337}"
export LAST_N_BLOCKS="${LAST_N_BLOCKS:-12}"
export TRAIN_CNN="${TRAIN_CNN:-0}"
export CNN_LR_MULT="${CNN_LR_MULT:-0.1}"
export SEED="${SEED:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/base_stage2.sh" "$@"
