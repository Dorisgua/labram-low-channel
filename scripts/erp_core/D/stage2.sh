#!/usr/bin/env bash
set -euo pipefail

# Dynamic Stage 2 wrapper：加载 Stage 1 corrector，冻结 CNN/corrector，
# 然后复用与 A/N/O 相同的 ERP CORE 分类执行器。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-erp_core_D_stage2/p_miss_add_Lmissingmae_delta_bs64_lr5e4_epoch200_missingw2_chkbest}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-p_miss_add_Lmissingmae_delta_bs64_lr5e4_epoch200_missingw2_chkbest}"
export MODEL="${MODEL:-labram_dynamic_base_patch200_200}"

# export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/erpcore/erp_core_D_stage1/p_miss_add_delta_bs256_lr1e4_epoch200/checkpoint-119.pth}"
# export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/erpcore/erp_core_D_stage1/p_miss_add_Lmissingmae_delta_bs256_lr1e4_epoch200/checkpoint-best.pth}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_DIR}/outputs/erpcore/erp_core_D_stage1/p_miss_add_Lmissingmae_delta_bs64_lr5e4_epoch200_missingw2/checkpoint-best.pth}"
export FINETUNE="${FINETUNE:-${STAGE1_CHECKPOINT}}"

export CHANNEL_SUBSET="erpcore12"
export COMPLETION_SCOPE="erpcore12_with_erpcore28"
export POOLING_SCOPE="high"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth}"
export CORRECTION_SCALE="${CORRECTION_SCALE:-1}"

export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="real"
export FREEZE_CNN="1"
export BEST_METRIC="${BEST_METRIC:-balanced_accuracy}" #balanced_accuracy

export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-4}"
export EPOCHS="${EPOCHS:-30}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEED="${SEED:-0}"
export MASTER_PORT="${MASTER_PORT:-29564}"

# 兼容旧 D 脚本的 RUN_BACKGROUND；新接口与 A/N/O 一致，使用 RUN_FOREGROUND。
if [[ -n "${RUN_BACKGROUND+x}" && -z "${RUN_FOREGROUND+x}" ]]; then
    case "${RUN_BACKGROUND}" in
        0) export RUN_FOREGROUND=1 ;;
        1) export RUN_FOREGROUND=0 ;;
        *) echo "RUN_BACKGROUND must be 0 or 1, got: ${RUN_BACKGROUND}" >&2; exit 2 ;;
    esac
fi

# Stage 1 checkpoint 的 key 不带 student. 前缀，因此必须关闭默认 gzp 过滤。
exec bash "${SCRIPT_DIR}/../base.sh" --model_filter_name "" "$@"
