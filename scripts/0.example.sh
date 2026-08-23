#!/usr/bin/env bash
# Example finetuning script for preexp12.
# This file is a parameter dictionary/template. Copy it to a new script and edit values.
# It is not meant to be run unchanged.

set -euo pipefail

# -----------------------------
# 0. Runtime
# -----------------------------

# Which GPUs to use.
# Choices/examples:
#   "0"       -> use GPU 0
#   "1"       -> use GPU 1
#   "0,1"     -> use GPU 0 and 1
#   "0,1,2,3" -> use four GPUs
CUDA_VISIBLE_DEVICES="0"

# Number of distributed processes on this node.
# Usually equals the number of GPUs in CUDA_VISIBLE_DEVICES.
# Choices/examples: 1, 2, 4, 8
NPROC_PER_NODE=1

# Limit CPU threads per process.
# Choices/examples: 1, 2, 4
OMP_NUM_THREADS=1

# Python/torchrun executable.
# Choices/examples:
#   torchrun
#   python -m torch.distributed.run
#   /path/to/env/bin/torchrun
TORCHRUN="torchrun"

# -----------------------------
# 1. Output
# -----------------------------

# Experiment name. Used to build output/log folders.
EXP_NAME="example_preexp12"

# Where checkpoints are saved.
# Recommended pattern: ./outputs/preexp12/checkpoints/${EXP_NAME}
OUTPUT_DIR="./outputs/tuev/preexp12/checkpoints/${EXP_NAME}"

# Where tensorboard logs are saved.
# Recommended pattern: ./outputs/preexp12/tensorboard/${EXP_NAME}
LOG_DIR="./outputs/preexp12/tensorboard/${EXP_NAME}"

# Where terminal output is saved.
# The final script can write the command at the top of this log, then append nohup output.
TERMINAL_LOG_DIR="./outputs/preexp12/run_logs"
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${EXP_NAME}.log"

# -----------------------------
# 2. Model and checkpoint
# -----------------------------

# Model architecture.
# Common choice:
#   labram_base_patch200_200
MODEL="labram_base_patch200_200"

# Pretrained checkpoint for finetuning.
# Choices/examples:
#   "./checkpoints/labram-base.pth" -> start from LaBraM base
#   ""                              -> train without finetune checkpoint
FINETUNE="./checkpoints/labram-base.pth"

# Checkpoint key used when loading FINETUNE.
# Common choices:
#   "model|module"
#   "model"
#   "module"
MODEL_KEY="model|module"

# Optional prefix to strip/add while loading checkpoint.
# Usually keep empty.【如果要加，是长什么样子的？】
MODEL_PREFIX=""

# Filter name used by the original LaBraM loader.
# Common choice: gzp【什么意思？】
MODEL_FILTER_NAME="gzp"

# Resume from a training checkpoint.
# Choices/examples:
#   "" -> do not resume
#   "./outputs/.../checkpoint-best.pth"
RESUME=""

# Auto resume from OUTPUT_DIR if checkpoint exists.
# Choices:
#   true  -> pass --auto_resume
#   false -> pass --no_auto_resume
AUTO_RESUME=false

# Evaluation only.
# Choices:
#   true  -> add --eval
#   false -> train normally
EVAL_ONLY=false

# -----------------------------
# 3. Dataset and channel subset
# -----------------------------

# Dataset name.
# Expected choices depend on run_class_finetuning.py DATASET_CONFIGS.
# Common choices:
#   TUEV
#   TUAB
#   SEEDV
#   AAD
DATASET="TUEV"

# Channel subset used by the dataset loader.
# Common choices/examples:
#   TUEV:  tuev13, tuev23
#   SEEDV: seedv23, seedv62
#   TUAB:  depends on utils.py/DATASET_CONFIGS
#   AAD:   depends on utils.py/DATASET_CONFIGS
CHANNEL_SUBSET="tuev13"

# Number of classes.
# Usually leave 0 and let DATASET_CONFIGS set it.
NB_CLASSES=0

# -----------------------------
# 4. Channel completion / prototype
# -----------------------------

# Completion scope: target channel space to complete into.
# Recommended choices for preexp12:
#   none                         -> no channel completion
#   tuev13_with_tuev23           -> input is TUEV subset, complete to TUEV-23
#   seedv23_with_seedv62         -> input is SEEDV subset, complete to SEEDV-62
#   tuev23_with_seedv62_extra    -> TUEV-23 plus SEEDV-extra target, total 70 channels
COMPLETION_SCOPE="tuev13_with_tuev23"

# Pooling scope after completion.
# Choices:
#   low  -> final pooling uses only real input-channel tokens
#   high -> final pooling uses all completed tokens, including prototype tokens
POOLING_SCOPE="high"

# Freeze CNN/patch_embed.
# Choices:
#   true  -> add --freeze_cnn; train transformer/head only
#   false -> train CNN/patch_embed as well
FREEZE_CNN=true

# Channel prototype path.
# Expected checkpoint key: channel_prototypes
# Expected shapes/examples:
#   tuev13_with_tuev23        -> [23, embed_dim]
#   seedv23_with_seedv62      -> [62, embed_dim]
#   tuev23_with_seedv62_extra -> source [62, embed_dim] or final [70, embed_dim], depending on implementation
CHANNEL_PROTOTYPE_PATH="docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth"

# -----------------------------
# 5. Optimization
# -----------------------------

# Batch size per process/GPU.
# Choices/examples: 16, 32, 64, 128
BATCH_SIZE=64

# Gradient accumulation steps.
# Effective batch size = BATCH_SIZE * NPROC_PER_NODE * UPDATE_FREQ.
# Choices/examples: 1, 2, 4, 8, 16
UPDATE_FREQ=8

# Learning rate.
# Common choices/examples: 1e-4, 3e-4, 5e-4, 1e-3
LR="5e-4"

# Number of epochs.
# Choices/examples: 30, 50, 100
EPOCHS=50

# Warmup epochs.
# Choices/examples: 0, 3, 5, 10
WARMUP_EPOCHS=5

# Weight decay.
# Common choices/examples: 0.01, 0.05, 0.1
WEIGHT_DECAY=0.05

# Layer-wise learning-rate decay.
# Common choices/examples: 0.65, 0.75, 0.8, 1.0
LAYER_DECAY=0.65

# Drop path rate.
# Common choices/examples: 0.0, 0.1, 0.2
DROP_PATH=0.1

# Label smoothing.
# Common choices/examples: 0.0, 0.1
SMOOTHING=0.1

# Optimizer.
# Common choices from timm/LaBraM setup:
#   adamw
OPT="adamw"

# -----------------------------
# 6. Model flags
# -----------------------------

# Relative position bias.
# Choices:
#   true  -> use rel_pos_bias if model supports it
#   false -> add --disable_rel_pos_bias
USE_REL_POS_BIAS=false

# Absolute position embedding.
# Choices:
#   true  -> add --abs_pos_emb
#   false -> do not add --abs_pos_emb
USE_ABS_POS_EMB=true

# QKV bias.
# Choices:
#   true  -> use qkv bias
#   false -> add --disable_qkv_bias
USE_QKV_BIAS=false

# Mean pooling vs cls token.
# Choices:
#   mean -> add/use --use_mean_pooling
#   cls  -> add --use_cls
POOLING_HEAD="mean"

# -----------------------------
# 7. Checkpoint/logging/data-loader
# -----------------------------

# Save checkpoint every N epochs.
# Choices/examples: 1, 5, 10
SAVE_CKPT_FREQ=5

# Save checkpoints.
# Choices:
#   true  -> save checkpoints
#   false -> add --no_save_ckpt if implemented; otherwise keep output_dir empty
SAVE_CKPT=true

# Number of data loader workers.
# Choices/examples: 0, 2, 4, 8, 16
NUM_WORKERS=4

# Pin memory in DataLoader.
# Choices:
#   true  -> add --pin_mem
#   false -> add --no_pin_mem if implemented
PIN_MEM=true

# Random seed.
# Choices/examples: 0, 1, 2, 42
SEED=0

# Device.
# Choices/examples:
#   cuda
#   cpu
DEVICE="cuda"

# -----------------------------
# 8. Command assembly
# -----------------------------

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --model "${MODEL}"
  --finetune "${FINETUNE}"
  --model_key "${MODEL_KEY}"
  --model_prefix "${MODEL_PREFIX}"
  --model_filter_name "${MODEL_FILTER_NAME}"
  --dataset "${DATASET}"
  --channel_subset "${CHANNEL_SUBSET}"
  --nb_classes "${NB_CLASSES}"
  --completion_scope "${COMPLETION_SCOPE}"
  --pooling_scope "${POOLING_SCOPE}"
  --channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}"
  --batch_size "${BATCH_SIZE}"
  --update_freq "${UPDATE_FREQ}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --weight_decay "${WEIGHT_DECAY}"
  --layer_decay "${LAYER_DECAY}"
  --drop_path "${DROP_PATH}"
  --smoothing "${SMOOTHING}"
  --opt "${OPT}"
  --save_ckpt_freq "${SAVE_CKPT_FREQ}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${FREEZE_CNN}" == "true" ]]; then
  ARGS+=(--freeze_cnn)
fi

if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

if [[ "${AUTO_RESUME}" == "true" ]]; then
  ARGS+=(--auto_resume)
else
  ARGS+=(--no_auto_resume)
fi

if [[ "${EVAL_ONLY}" == "true" ]]; then
  ARGS+=(--eval)
fi

if [[ "${USE_REL_POS_BIAS}" == "false" ]]; then
  ARGS+=(--disable_rel_pos_bias)
fi

if [[ "${USE_ABS_POS_EMB}" == "true" ]]; then
  ARGS+=(--abs_pos_emb)
fi

if [[ "${USE_QKV_BIAS}" == "false" ]]; then
  ARGS+=(--disable_qkv_bias)
fi

if [[ "${POOLING_HEAD}" == "cls" ]]; then
  ARGS+=(--use_cls)
else
  ARGS+=(--use_mean_pooling)
fi

# Uncomment after copying this template to a real experiment script.
# mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TERMINAL_LOG_DIR}"
# CMD=(
#   "${TORCHRUN}"
#   --nnodes=1
#   --nproc_per_node="${NPROC_PER_NODE}"
#   run_class_finetuning.py
#   "${ARGS[@]}"
# )
#
# {
#   echo "Command:"
#   printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
#   printf '%q ' "${CMD[@]}"
#   printf '\n'
#   echo
#   echo "Output:"
# } > "${TERMINAL_LOG}"
#
# nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &

printf '%q ' CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${TORCHRUN}" --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" run_class_finetuning.py "${ARGS[@]}"
printf '\n'
