#!/usr/bin/env bash
set -euo pipefail
# LaBraM base on TUEV-23: freeze CNN/patch_embed and train transformer/head.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
RUN_PREFIX="${SCRIPT_NAME%.sh}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_PREFIX}_${RUN_ID}"

# GPU choices/examples:
#   "0"       -> use GPU 0
#   "1"       -> use GPU 1
#   "0,1"     -> use GPU 0 and GPU 1
#   "0,1,2,3" -> use four GPUs
GPU_IDS="${GPU_IDS:-0}"

# Number of distributed processes on this node.
# Choices/examples:
#   1 -> single GPU/process
#   2 -> two GPUs/processes
#   4 -> four GPUs/processes
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Batch size per GPU/process.
BATCH_SIZE=64

# Gradient accumulation steps.
# Effective batch size = BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE.
UPDATE_FREQ=8

# Learning rate.
LR="5e-4"

# Number of training epochs.
EPOCHS=50

# Number of warmup epochs.
WARMUP_EPOCHS=5

# Weight decay.
WEIGHT_DECAY=0.05

# Layer-wise learning-rate decay.
# Common choices/examples:
#   1.0  -> no layer-wise decay
#   0.75 -> mild layer-wise decay
#   0.65 -> stronger layer-wise decay
LAYER_DECAY=0.65

# Drop path rate.
DROP_PATH=0.1

# Save checkpoint every N epochs.
SAVE_CKPT_FREQ=5

# DataLoader workers.
NUM_WORKERS="${NUM_WORKERS:-4}"

# Random seed.
SEED=0

# Model choices/examples:
#   labram_base_patch200_200
#   labram_large_patch200_200
#   labram_huge_patch200_200
MODEL="labram_base_patch200_200"

# Finetune checkpoint path.
FINETUNE="./checkpoints/labram-base.pth"

# Dataset choices currently used in this repo:
#   TUEV
#   TUAB
DATASET="TUEV"

# Channel subset choices currently planned:
#   TUEV:  tuev13, tuev23
#   SEEDV: seedv23, seedv62
CHANNEL_SUBSET="tuev13"

# Channel completion choices:
#   none                         -> no channel completion
#   tuev13_with_tuev23           -> input TUEV-13, complete to TUEV-23
#   seedv23_with_seedv62         -> input SEEDV-23, complete to SEEDV-62
#   tuev23_with_seedv62_extra    -> input TUEV-23, complete to TUEV-23 + SEEDV-extra
COMPLETION_SCOPE="tuev13_with_tuev23"
CHANNEL_PROTOTYPE_PATH="docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth"

# Pooling choices:
#   low  -> pool only real input-channel tokens
#   high -> pool all target-channel tokens after completion
POOLING_SCOPE="high"

# Validation metric used to select checkpoint-best.
# Choices:
#   accuracy
#   balanced_accuracy
#   f1_weighted
#   cohen_kappa
#   roc_auc
#   pr_auc
# Recommended:
#   TUEV -> cohen_kappa
#   TUAB -> roc_auc
#   SEEDV -> accuracy
BEST_METRIC="cohen_kappa"

# Freeze CNN/patch_embed switch.
# Choices:
#   1 -> pass --freeze_cnn and train only transformer/head layers
#   0 -> do not pass --freeze_cnn and finetune CNN/patch_embed as well
FREEZE_CNN="${FREEZE_CNN:-1}"

# Resume checkpoint path.
# Choices/examples:
#   "" -> do not resume
#   "./outputs/.../checkpoint-best.pth" -> resume from checkpoint
RESUME="${RESUME:-}"

# Evaluation-only switch.
# Choices:
#   0 -> train
#   1 -> evaluate only
EVAL_ONLY="${EVAL_ONLY:-0}"

# Auto-resume switch.
# Choices:
#   1 -> pass --no_auto_resume
#   0 -> allow default auto-resume behavior
NO_AUTO_RESUME=1

# Label smoothing.
# Choices/examples:
#   ""    -> do not pass --smoothing
#   "0.0" -> no smoothing
#   "0.1" -> common smoothing value
SMOOTHING="0.1"

# torchrun executable path.
TORCHRUN="/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun"

# CUDA visible devices. Defaults to GPU_IDS unless already set outside the script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

# CPU threads per process.
# Choices/examples:
#   1, 2, 4, 8
OMP_NUM_THREADS=1

# Experiment group name used under ./outputs/.
EXP_GROUP="preexp17_tuev"
OUTPUT_ROOT="./outputs/tuev/${EXP_GROUP}"
OUTPUT_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}/"
TB_LOG_DIR="${OUTPUT_ROOT}/tensorboard/${RUN_NAME}/"
TERMINAL_LOG_DIR="${OUTPUT_ROOT}/run_logs"
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${RUN_NAME}.log"

if [[ ! -f "${FINETUNE}" ]]; then
    echo "Missing finetune checkpoint: ${FINETUNE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${TB_LOG_DIR}" "${TERMINAL_LOG_DIR}"

CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    run_class_finetuning.py
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${TB_LOG_DIR}"
    --model "${MODEL}"
    --finetune "${FINETUNE}"
    --dataset "${DATASET}"
    --channel_subset "${CHANNEL_SUBSET}"
    --completion_scope "${COMPLETION_SCOPE}"
    --pooling_scope "${POOLING_SCOPE}"
    --best_metric "${BEST_METRIC}"
    --batch_size "${BATCH_SIZE}"
    --update_freq "${UPDATE_FREQ}"
    --lr "${LR}"
    --epochs "${EPOCHS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --weight_decay "${WEIGHT_DECAY}"
    --layer_decay "${LAYER_DECAY}"
    --drop_path "${DROP_PATH}"
    --save_ckpt_freq "${SAVE_CKPT_FREQ}"
    --disable_rel_pos_bias
    --abs_pos_emb
    --disable_qkv_bias
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
)
if [[ -n "${CHANNEL_PROTOTYPE_PATH}" ]]; then
      CMD+=(--channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}")
fi

if [[ "${FREEZE_CNN}" == "1" ]]; then
    CMD+=(--freeze_cnn)
fi

if [[ -n "${SMOOTHING}" ]]; then
    CMD+=(--smoothing "${SMOOTHING}")
fi

if [[ -n "${RESUME}" ]]; then
    CMD+=(--resume "${RESUME}")
fi

if [[ "${EVAL_ONLY}" == "1" ]]; then
    CMD+=(--eval)
fi

if [[ "${NO_AUTO_RESUME}" == "1" ]]; then
    CMD+=(--no_auto_resume)
fi

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    echo
    echo "Output:"
} > "${TERMINAL_LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &
echo "Started ${RUN_NAME}"
echo "Log: ${TERMINAL_LOG}"
echo "Global batch size: $((BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE))"
