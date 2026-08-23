#!/usr/bin/env bash
set -euo pipefail
# BCI-IV-2a 33A: input 13 real channels, complete to 22, freeze CNN,
# and train the Transformer plus AdaBrain all-token classification head.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_PREFIX}_${RUN_ID}"

# Runtime. NPROC_PER_NODE should match the number of visible GPUs.
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# AdaBrain-aligned BCI-IV-2a optimization settings.
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
SMOOTHING="${SMOOTHING:-0.1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
SEED="${SEED:-0}"

MODEL="labram_base_patch200_200"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATASET="bciiv2a"
CHANNEL_SUBSET="bciiv2a13"
DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/AdaBrain-PreExp34-35-repro/AdaBrain-Bench-main_film/preprocessing/BCI-IV-2A/multi_subject_json}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
NORM_METHOD="${NORM_METHOD:-z_score}"
BEST_METRIC="${BEST_METRIC:-balanced_accuracy}"
CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"

# Complete the 13 real channels to the original BCI-IV-2a 22-channel space.
COMPLETION_SCOPE="bciiv2a13_with_bciiv2a22"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_bciiv2a22_cnn_patch_embed_mean.pth}"
POOLING_SCOPE="high"

RESUME="${RESUME:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"

TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

EXP_GROUP="preexp33_bciiv2a_multisession"
OUTPUT_ROOT="./outputs/bciiv2a/${EXP_GROUP}"
OUTPUT_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}/"
TB_LOG_DIR="${OUTPUT_ROOT}/tensorboard/${RUN_NAME}/"
TERMINAL_LOG_DIR="${OUTPUT_ROOT}/run_logs"
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${RUN_NAME}.log"

if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing torchrun executable: ${TORCHRUN}"
    exit 1
fi
if [[ ! -f "${FINETUNE}" ]]; then
    echo "Missing finetune checkpoint: ${FINETUNE}"
    exit 1
fi
if [[ ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing BCI-IV-2a channel prototype: ${CHANNEL_PROTOTYPE_PATH}"
    echo "Generate it with: python docs/prototypes/01_generate_bciiv2a_cnn_patch_prototypes.py"
    exit 1
fi
for split in train val test; do
    if [[ ! -f "${DATA_PATH}/${split}.json" ]]; then
        echo "Missing BCI-IV-2a split manifest: ${DATA_PATH}/${split}.json"
        exit 1
    fi
done

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
    --data_path "${DATA_PATH}"
    --sampling_rate "${SAMPLING_RATE}"
    --norm_method "${NORM_METHOD}"
    --completion_scope "${COMPLETION_SCOPE}"
    --channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}"
    --pooling_scope "${POOLING_SCOPE}"
    --classifier_mode adabrain_all_token
    --classifier_token_scope "${CLASSIFIER_TOKEN_SCOPE}"
    --best_metric "${BEST_METRIC}"
    --batch_size "${BATCH_SIZE}"
    --update_freq "${UPDATE_FREQ}"
    --lr "${LR}"
    --epochs "${EPOCHS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --weight_decay "${WEIGHT_DECAY}"
    --layer_decay "${LAYER_DECAY}"
    --drop_path "${DROP_PATH}"
    --smoothing "${SMOOTHING}"
    --save_ckpt_freq "${SAVE_CKPT_FREQ}"
    --abs_pos_emb
    --freeze_cnn
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
)

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

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &

echo "Started ${RUN_NAME}"
echo "Log: ${TERMINAL_LOG}"
echo "Global batch size: $((BATCH_SIZE * UPDATE_FREQ * NPROC_PER_NODE))"
