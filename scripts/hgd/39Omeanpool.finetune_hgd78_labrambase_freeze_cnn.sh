#!/usr/bin/env bash
set -euo pipefail
# PreExp39 HGD 基线：使用 78 个 LaBraM 可定位通道，保留官方 test 文件，
# 不做通道补全，冻结 CNN/patch_embed，分类头使用 LaBraM mean pooling。

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# 运行命名。
# RUN_PREFIX_OVERRIDE：wrapper 脚本可以设置它，让输出名字使用 wrapper 的名字，
# 而不是这个 base launcher 的名字。
# RUN_TAG：可选的附加标签，会插到时间戳前面；批量跑任务时可用，例如 seed0_task01。
RUN_PREFIX="${RUN_PREFIX_OVERRIDE:-${SCRIPT_NAME%.sh}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-}"
RUN_NAME="${RUN_PREFIX}${RUN_TAG:+_${RUN_TAG}}_${RUN_ID}"

# 运行资源。
# GPU_IDS：暴露给任务的物理 GPU 编号，例如 0 或 0,1。
# CUDA_VISIBLE_DEVICES：默认等于 GPU_IDS；只有需要特殊映射时才单独覆盖。
# NPROC_PER_NODE：torchrun worker 数；当前单卡实验通常设为 1。
# MASTER_PORT：torch 分布式通信端口；同一机器并行跑多个任务时需要换端口。
# RUN_FOREGROUND：0 表示用 nohup 后台启动；1 表示前台运行并实时打印训练日志。
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29519}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"

# 优化参数默认值。
# BATCH_SIZE：每个进程的 batch size；全局 batch = BATCH_SIZE * UPDATE_FREQ *
# NPROC_PER_NODE。
# UPDATE_FREQ：梯度累积步数。
# LR：finetune 基础学习率。
# EPOCHS：总训练轮数。这里默认 2，适合 smoke run；正式跑可设 EPOCHS=50。
# WARMUP_EPOCHS：学习率 warmup 轮数。
# WEIGHT_DECAY：AdamW 的 weight decay。
# LAYER_DECAY：逐层学习率衰减；1.0 表示不使用逐层衰减。
# DROP_PATH：stochastic depth 概率。
# SMOOTHING：label smoothing。
# SAVE_CKPT_FREQ：每隔多少个 epoch 存一次 checkpoint。
# SEED：训练随机种子。
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

# 模型和数据。
# MODEL：modeling 里注册的 LaBraM 模型结构。
# FINETUNE：要加载的预训练 checkpoint。
# DATA_PATH：dataset_maker/make_HGD.py 的输出目录，里面必须有 manifest.json。
# CHANNEL_SUBSET 可选值：
#   hgd78 = 78 个 LaBraM 可定位 HGD 通道
#   hgd20 = 20 个运动皮层通道子集，39N wrapper 会使用这个选项
# COMPLETION_SCOPE：
#   none = 不补通道
#   hgd20_with_hgd78 = 20 个真实 HGD 通道补全到 HGD-78，39A wrapper 会使用
# CHANNEL_PROTOTYPE_PATH：completion_scope 非 none 时必须指向 prototype .pth。
# TORCHRUN：labram 环境里的 torchrun 可执行文件。
MODEL="labram_base_patch200_200"
FINETUNE="${FINETUNE:-./checkpoints/labram-base.pth}"
DATA_PATH="${DATA_PATH:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/HGD/processed_data_4s_200hz}"
CHANNEL_SUBSET="${CHANNEL_SUBSET:-hgd78}"
COMPLETION_SCOPE="${COMPLETION_SCOPE:-none}"
POOLING_SCOPE="${POOLING_SCOPE:-low}"
CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-}"
TORCHRUN="${TORCHRUN:-/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun}"

# 输出目录。
# EXP_GROUP：控制 outputs/ 下面的实验组目录。
# OUTPUT_DIR：保存 checkpoint 和每个 epoch 的 JSONL 日志。
# TB_LOG_DIR：保存 TensorBoard 日志。
# TERMINAL_LOG：保存 shell 和 torchrun 的 stdout/stderr。
EXP_GROUP="${EXP_GROUP:-preexp39_hgd78_mean_pool_official_split}"
OUTPUT_ROOT="./outputs/hgd/${EXP_GROUP}"
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
if [[ ! -f "${DATA_PATH}/manifest.json" ]]; then
    echo "Missing HGD manifest: ${DATA_PATH}/manifest.json"
    echo "Run: python dataset_maker/make_HGD.py"
    exit 1
fi
if [[ "${COMPLETION_SCOPE}" != "none" && ! -f "${CHANNEL_PROTOTYPE_PATH}" ]]; then
    echo "Missing HGD channel prototype: ${CHANNEL_PROTOTYPE_PATH}"
    echo "Run: python docs/prototypes/01_generate_hgd_cnn_patch_prototypes.py"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${TB_LOG_DIR}" "${TERMINAL_LOG_DIR}"

CMD=(
    "${TORCHRUN}"
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    run_class_finetuning.py
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${TB_LOG_DIR}"
    --model "${MODEL}"
    --finetune "${FINETUNE}"
    # 这个 launcher 固定的数据设置：
    # --dataset HGD 会选择 data_processor/hgd.py。
    # --channel_subset 默认是 hgd78，也可以被 wrapper 覆盖为 hgd20。
    # --sampling_rate 200 对应已经预处理好的 HGD 数组。
    # --norm_method z_score 表示用训练集统计量做通道级 z-score。
    --dataset HGD
    --channel_subset "${CHANNEL_SUBSET}"
    --data_path "${DATA_PATH}"
    --sampling_rate 200
    --norm_method z_score
    # completion_scope=none 表示不补通道；wrapper 可以覆盖为 hgd20_with_hgd78。
    # pooling_scope=low 表示分类池化只使用真实输入通道；high 表示池化补全后的全部 target 通道。
    --completion_scope "${COMPLETION_SCOPE}"
    --pooling_scope "${POOLING_SCOPE}"
    --channel_prototype_path "${CHANNEL_PROTOTYPE_PATH}"
    # mean_pool 表示使用 LaBraM mean pooling 分类头，而不是 AdaBrain 的
    # flattened all-token 分类头。
    --classifier_mode mean_pool
    # 按 validation accuracy 选择 best checkpoint。
    --best_metric accuracy
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
    --disable_rel_pos_bias
    --abs_pos_emb
    --disable_qkv_bias
    --freeze_cnn
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --no_auto_resume
)

{
    echo "Command:"
    printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q ' "${CUDA_VISIBLE_DEVICES}" "${OMP_NUM_THREADS}"
    printf '%q ' "${CMD[@]}"
    printf '\n\n'
    echo "HGD: annotation onset 0-4 s; 200 Hz; channel_subset=${CHANNEL_SUBSET}"
    echo "Protocol: official train/test files; stratified 20% validation from train"
    echo "This protocol is not cross-subject; subjects 1-14 occur in every split"
    echo "Completion: ${COMPLETION_SCOPE}; prototype=${CHANNEL_PROTOTYPE_PATH:-<none>}; pooling_scope=${POOLING_SCOPE}"
    echo "Classifier: LaBraM mean pooling; CNN/patch_embed frozen"
    echo "Terminal log: ${TERMINAL_LOG}"
} | tee "${TERMINAL_LOG}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    echo "Starting ${RUN_NAME} in foreground"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" 2>&1 | tee -a "${TERMINAL_LOG}"
else
    nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        "${CMD[@]}" >> "${TERMINAL_LOG}" 2>&1 &
    echo "Started ${RUN_NAME} in background"
    echo "PID: $!"
    echo "Log: ${TERMINAL_LOG}"
fi
