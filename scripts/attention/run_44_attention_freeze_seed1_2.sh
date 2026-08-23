#!/usr/bin/env bash
set -euo pipefail

# Run Attention freeze-CNN configs for missing seeds.
# Defaults: 44O/44N/44A with seed 1 and seed 2.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

TASK_GPUS="${TASK_GPUS:-auto}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-1024}"
GPU_FREE_UTIL_MAX="${GPU_FREE_UTIL_MAX:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ATTENTION_SEEDS=(${ATTENTION_SEEDS:-1 2})

ATTENTION_O_SCRIPT="scripts/44Omeanpool.finetune_attention26_labrambase_freeze_cnn.sh"
ATTENTION_N_SCRIPT="scripts/44Nmeanpool.finetune_attention10_labrambase_freeze_cnn.sh"
ATTENTION_A_SCRIPT="scripts/44Ameanpool.finetune_attention10_with_attention26_prototype_labrambase_freeze_cnn.sh"
ATTENTION_PROTOTYPE="docs/prototypes/01_attention26_cnn_patch_embed_mean.pth"
ATTENTION_MANIFEST="/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Attention/processed_data_4s_200hz/manifest.json"

for script in "${ATTENTION_O_SCRIPT}" "${ATTENTION_N_SCRIPT}" "${ATTENTION_A_SCRIPT}"; do
    if [[ ! -x "${script}" ]]; then
        echo "Missing executable script: ${script}"
        exit 1
    fi
    bash -n "${script}"
done

if [[ ! -f "${ATTENTION_PROTOTYPE}" ]]; then
    echo "Missing Attention prototype: ${ATTENTION_PROTOTYPE}"
    exit 1
fi
if [[ ! -f "${ATTENTION_MANIFEST}" ]]; then
    echo "Missing Attention manifest: ${ATTENTION_MANIFEST}"
    exit 1
fi

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG_DIR="outputs/selected_batch_logs"
BATCH_LOG="${BATCH_LOG_DIR}/run_44_attention_freeze_seed1_2_${BATCH_ID}.log"
mkdir -p "${BATCH_LOG_DIR}"
exec > >(tee -a "${BATCH_LOG}") 2>&1

TASK_NAMES=()
TASK_SCRIPTS=()
TASK_SEEDS=()

add_task() {
    TASK_NAMES+=("$1")
    TASK_SCRIPTS+=("$2")
    TASK_SEEDS+=("$3")
}

for seed in "${ATTENTION_SEEDS[@]}"; do
    add_task "44O_attention26_freeze_seed${seed}" "${ATTENTION_O_SCRIPT}" "${seed}"
    add_task "44N_attention10_freeze_seed${seed}" "${ATTENTION_N_SCRIPT}" "${seed}"
    add_task "44A_attention10_proto_freeze_seed${seed}" "${ATTENTION_A_SCRIPT}" "${seed}"
done

total_runs="${#TASK_SCRIPTS[@]}"

run_one() {
    local task_index="$1"
    local gpu="$2"
    local display_index=$((task_index + 1))
    local name="${TASK_NAMES[task_index]}"
    local script="${TASK_SCRIPTS[task_index]}"
    local seed="${TASK_SEEDS[task_index]}"
    local port=$((29900 + task_index))
    local run_tag="seed${seed}_freeze$(printf '%02d' "${display_index}")"
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
        RUN_FOREGROUND=1
        SEED="${seed}"
        GPU_IDS="${gpu}"
        CUDA_VISIBLE_DEVICES="${gpu}"
        NPROC_PER_NODE="${NPROC_PER_NODE}"
        MASTER_PORT="${port}"
        RUN_TAG="${run_tag}"
        bash "${script}"
    )

    echo
    echo "[${display_index}/${total_runs}] gpu=${gpu} name=${name} seed=${seed}"
    echo "script=${script}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN: '
        printf '%q ' "${launch_cmd[@]}"
        printf '\n'
        return 0
    fi
    "${launch_cmd[@]}"
}

if [[ "${TASK_GPUS}" == "auto" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "TASK_GPUS=auto requires nvidia-smi"
        exit 1
    fi
    CANDIDATE_GPUS=()
    while IFS=',' read -r raw_index; do
        gpu_index="$(echo "${raw_index}" | tr -d '[:space:]')"
        [[ -n "${gpu_index}" ]] && CANDIDATE_GPUS+=("${gpu_index}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
    IFS=',' read -r -a CANDIDATE_GPUS <<< "${TASK_GPUS}"
fi

if [[ "${#CANDIDATE_GPUS[@]}" -lt 1 ]]; then
    echo "No candidate GPU found"
    exit 1
fi

gpu_is_reserved() {
    local gpu="$1"
    local reserved_gpu
    for reserved_gpu in "${RUNNING_GPUS[@]:-}"; do
        [[ "${reserved_gpu}" == "${gpu}" ]] && return 0
    done
    return 1
}

find_idle_gpu() {
    local raw_index raw_memory raw_util gpu_index memory_used gpu_util candidate
    while IFS=',' read -r raw_index raw_memory raw_util; do
        gpu_index="$(echo "${raw_index}" | tr -d '[:space:]')"
        memory_used="$(echo "${raw_memory}" | tr -d '[:space:]')"
        gpu_util="$(echo "${raw_util}" | tr -d '[:space:]')"
        for candidate in "${CANDIDATE_GPUS[@]}"; do
            [[ "${gpu_index}" != "${candidate}" ]] && continue
            gpu_is_reserved "${gpu_index}" && continue
            if (( memory_used <= GPU_FREE_MEMORY_MAX_MB && gpu_util <= GPU_FREE_UTIL_MAX )); then
                echo "${gpu_index}"
                return 0
            fi
        done
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 1
}

reap_finished_tasks() {
    local new_pids=()
    local new_gpus=()
    local running_jobs pid status i
    running_jobs="$(jobs -pr)"
    for i in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[i]}"
        if printf '%s\n' "${running_jobs}" | grep -qx "${pid}"; then
            new_pids+=("${pid}")
            new_gpus+=("${RUNNING_GPUS[i]}")
            continue
        fi
        if wait "${pid}"; then
            status=0
        else
            status="$?"
        fi
        if [[ "${status}" != "0" ]]; then
            echo "Task pid=${pid} on gpu=${RUNNING_GPUS[i]} failed with status ${status}"
            parallel_status=1
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
    RUNNING_GPUS=("${new_gpus[@]}")
}

echo "Attention freeze-CNN seed1/2 batch"
echo "Batch id: ${BATCH_ID}"
echo "Batch log: ${BATCH_LOG}"
echo "Candidate GPUs: ${CANDIDATE_GPUS[*]}"
echo "GPU idle thresholds: memory.used <= ${GPU_FREE_MEMORY_MAX_MB} MiB, utilization.gpu <= ${GPU_FREE_UTIL_MAX}%"
echo "GPU poll seconds: ${GPU_POLL_SECONDS}"
echo "Attention seeds: ${ATTENTION_SEEDS[*]}"
echo "Total runs: ${total_runs}"
echo "DRY_RUN: ${DRY_RUN}"

parallel_status=0
RUNNING_PIDS=()
RUNNING_GPUS=()
next_task=0

while (( next_task < total_runs || ${#RUNNING_PIDS[@]} > 0 )); do
    reap_finished_tasks
    launched_any=0
    while (( next_task < total_runs )); do
        if idle_gpu="$(find_idle_gpu)"; then
            run_one "${next_task}" "${idle_gpu}" &
            RUNNING_PIDS+=("$!")
            RUNNING_GPUS+=("${idle_gpu}")
            next_task=$((next_task + 1))
            launched_any=1
        else
            break
        fi
    done
    if (( next_task >= total_runs )); then
        sleep 5
        continue
    fi
    if (( launched_any == 0 )); then
        echo
        echo "No idle candidate GPU now; waiting ${GPU_POLL_SECONDS}s before checking again."
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
        sleep "${GPU_POLL_SECONDS}"
    fi
done

if [[ "${parallel_status}" != "0" ]]; then
    echo "At least one Attention freeze task failed; inspect ${BATCH_LOG} and per-run logs"
    exit "${parallel_status}"
fi

echo
echo "Completed all ${total_runs} Attention freeze runs"
echo "Batch log: ${BATCH_LOG}"
