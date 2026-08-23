#!/usr/bin/env bash
set -euo pipefail
# Run the six PreExp36 SEED configurations for seeds 0, 1, and 2.
# Default: sequential execution. Set TASK_GPUS=0,1 to run two independent
# single-GPU tasks concurrently, one worker per physical GPU.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TASK_GPUS="${TASK_GPUS:-}"
RUN_FOREGROUND=1
DRY_RUN="${DRY_RUN:-0}"

SEED_PROTOTYPE="docs/prototypes/01_seed62_cnn_patch_embed_mean.pth"
EXPERIMENTS=(
    "scripts/36Oada.finetune_seed62_labrambase_freeze_cnn.sh"
    "scripts/36Nada.finetune_seed23_labrambase_freeze_cnn.sh"
    "scripts/36Omeanpool.finetune_seed62_labrambase_freeze_cnn.sh"
    "scripts/36Nmeanpool.finetune_seed23_labrambase_freeze_cnn.sh"
    "scripts/36Ahmeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh"
    "scripts/36Almeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh"
)
SEEDS=(0 1 2)

for experiment in "${EXPERIMENTS[@]}"; do
    if [[ ! -x "${experiment}" ]]; then
        echo "Missing executable experiment launcher: ${experiment}"
        exit 1
    fi
    bash -n "${experiment}"
done

if [[ "${DRY_RUN}" != "1" && ! -f "${SEED_PROTOTYPE}" ]]; then
    echo "Missing SEED prototype required by 36Ah/36Al: ${SEED_PROTOTYPE}"
    echo "Generate it with:"
    echo "  CUDA_VISIBLE_DEVICES=1 python docs/prototypes/01_generate_seed_cnn_patch_prototypes.py"
    exit 1
fi

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG_DIR="outputs/preexp36_batch_logs"
BATCH_LOG="${BATCH_LOG_DIR}/run_36_experiments_seed012_${BATCH_ID}.log"
mkdir -p "${BATCH_LOG_DIR}"

exec > >(tee -a "${BATCH_LOG}") 2>&1

TASK_EXPERIMENTS=()
TASK_SEEDS=()
for experiment in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_EXPERIMENTS+=("${experiment}")
        TASK_SEEDS+=("${seed}")
    done
done
total_runs="${#TASK_EXPERIMENTS[@]}"

run_one() {
    local task_index="$1"
    local experiment="$2"
    local seed="$3"
    local task_gpu_ids="$4"
    local task_nproc="$5"
    local task_port="$6"
    local worker_label="$7"
    local display_index=$((task_index + 1))
    local run_tag="seed${seed}_task$(printf '%02d' "${display_index}")"
    local -a launch_cmd=(
        env
        -u CHANNEL_SUBSET
        -u CLASSIFIER_MODE
        -u COMPLETION_SCOPE
        -u CHANNEL_PROTOTYPE_PATH
        -u POOLING_SCOPE
        -u CLASSIFIER_TOKEN_SCOPE
        -u EXP_GROUP
        -u RUN_PREFIX_OVERRIDE
        -u MASTER_PORT
        -u RUN_TAG
        RUN_FOREGROUND="${RUN_FOREGROUND}"
        SEED="${seed}"
        GPU_IDS="${task_gpu_ids}"
        CUDA_VISIBLE_DEVICES="${task_gpu_ids}"
        NPROC_PER_NODE="${task_nproc}"
        RUN_TAG="${run_tag}"
    )
    if [[ -n "${task_port}" ]]; then
        launch_cmd+=(MASTER_PORT="${task_port}")
    fi
    launch_cmd+=(bash "${experiment}")

    echo
    echo "[${display_index}/${total_runs}] worker=${worker_label} experiment=${experiment} seed=${seed}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN: '
        printf '%q ' "${launch_cmd[@]}"
        printf '\n'
        return 0
    fi
    "${launch_cmd[@]}"
}

echo "PreExp36 batch"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: ${total_runs}"
echo "Batch log: ${BATCH_LOG}"

if [[ -z "${TASK_GPUS}" ]]; then
    echo "Mode: sequential"
    echo "GPU_IDS: ${GPU_IDS}"
    echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
    for ((task_index = 0; task_index < total_runs; task_index++)); do
        run_one \
            "${task_index}" \
            "${TASK_EXPERIMENTS[task_index]}" \
            "${TASK_SEEDS[task_index]}" \
            "${GPU_IDS}" \
            "${NPROC_PER_NODE}" \
            "" \
            "sequential"
    done
else
    IFS=',' read -r -a TASK_GPU_LIST <<< "${TASK_GPUS}"
    if [[ "${#TASK_GPU_LIST[@]}" -lt 1 ]]; then
        echo "TASK_GPUS must contain at least one GPU index"
        exit 1
    fi
    for gpu in "${TASK_GPU_LIST[@]}"; do
        if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU index in TASK_GPUS: ${gpu}"
            exit 1
        fi
    done

    worker_count="${#TASK_GPU_LIST[@]}"
    echo "Mode: task-parallel"
    echo "TASK_GPUS: ${TASK_GPU_LIST[*]}"
    echo "Workers: ${worker_count}"
    echo "NPROC_PER_NODE per task: 1"

    run_worker() {
        local worker_index="$1"
        local gpu="$2"
        local task_index
        for ((task_index = worker_index; task_index < total_runs; task_index += worker_count)); do
            run_one \
                "${task_index}" \
                "${TASK_EXPERIMENTS[task_index]}" \
                "${TASK_SEEDS[task_index]}" \
                "${gpu}" \
                "1" \
                "$((29600 + task_index))" \
                "gpu${gpu}"
        done
    }

    worker_pids=()
    for worker_index in "${!TASK_GPU_LIST[@]}"; do
        run_worker "${worker_index}" "${TASK_GPU_LIST[worker_index]}" &
        worker_pids+=("$!")
    done

    parallel_status=0
    for worker_pid in "${worker_pids[@]}"; do
        if ! wait "${worker_pid}"; then
            parallel_status=1
        fi
    done
    if [[ "${parallel_status}" != "0" ]]; then
        echo "At least one GPU worker failed; inspect the per-run and batch logs"
        exit "${parallel_status}"
    fi
fi

echo
echo "Completed all ${total_runs} PreExp36 runs"
echo "Batch log: ${BATCH_LOG}"
