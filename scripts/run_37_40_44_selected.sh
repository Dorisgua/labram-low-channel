#!/usr/bin/env bash
set -euo pipefail

# Run selected EEGMAT, Siena, and Attention experiments.
#
# Default plan:
#   - EEGMAT 37Nada seed 0/1/2
#   - EEGMAT 37Aada seed 0/1/2
#   - Siena 40Aada seed 0
#   - Siena 40Nada seed 0
#   - Attention 44O full fine-tune seed 0
#   - Attention 44N 10-channel full fine-tune seed 0
#   - Attention 44A 10-channel + prototype full fine-tune seed 0
#
# Default execution checks GPUs before every new run. A GPU is considered idle
# when memory.used <= GPU_FREE_MEMORY_MAX_MB and utilization.gpu <= GPU_FREE_UTIL_MAX.
# Override the candidate GPU list with TASK_GPUS=0,1 or TASK_GPUS=1.
# Set DRY_RUN=1 to print commands only.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

TASK_GPUS="${TASK_GPUS:-auto}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-1024}"
GPU_FREE_UTIL_MAX="${GPU_FREE_UTIL_MAX:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FOREGROUND=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

EEGMAT_SEEDS=(${EEGMAT_SEEDS:-0 1 2})
SIENA_SEEDS=(${SIENA_SEEDS:-0})
ATTENTION_SEEDS=(${ATTENTION_SEEDS:-0})

EEGMAT_N_SCRIPT="scripts/eegmat/37Nada.finetune_eegmat8_labrambase_freeze_cnn.sh"
EEGMAT_A_SCRIPT="scripts/eegmat/37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn.sh"
SIENA_A_SCRIPT="scripts/siena/40Aada.finetune_siena13_with_siena29_prototype_labrambase_freeze_cnn.sh"
SIENA_N_SCRIPT="scripts/siena/40Nada.finetune_siena13_labrambase_freeze_cnn.sh"
ATTENTION_FULL_SCRIPT="scripts/attention/44Omeanpool.finetune_attention26_labrambase_full_finetune.sh"
ATTENTION_N_SCRIPT="scripts/attention/44Nmeanpool.finetune_attention10_labrambase_full_finetune.sh"
ATTENTION_A_SCRIPT="scripts/attention/44Ameanpool.finetune_attention10_with_attention26_prototype_labrambase_full_finetune.sh"

EEGMAT_PROTOTYPE="docs/prototypes/01_eegmat19_cnn_patch_embed_mean.pth"
SIENA_PROTOTYPE="docs/prototypes/01_siena29_cnn_patch_embed_mean.pth"
ATTENTION_PROTOTYPE="docs/prototypes/01_attention26_cnn_patch_embed_mean.pth"
ATTENTION_MANIFEST="/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Attention/processed_data_4s_200hz/manifest.json"

REQUIRED_SCRIPTS=(
    "${EEGMAT_N_SCRIPT}"
    "${EEGMAT_A_SCRIPT}"
    "${SIENA_A_SCRIPT}"
    "${SIENA_N_SCRIPT}"
    "${ATTENTION_FULL_SCRIPT}"
    "${ATTENTION_N_SCRIPT}"
    "${ATTENTION_A_SCRIPT}"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -x "${script}" ]]; then
        echo "Missing executable script: ${script}"
        exit 1
    fi
    bash -n "${script}"
done

if [[ ! -f "${EEGMAT_PROTOTYPE}" ]]; then
    echo "Missing EEGMAT prototype: ${EEGMAT_PROTOTYPE}"
    exit 1
fi
if [[ ! -f "${SIENA_PROTOTYPE}" ]]; then
    echo "Missing Siena prototype: ${SIENA_PROTOTYPE}"
    exit 1
fi
if [[ ! -f "${ATTENTION_PROTOTYPE}" ]]; then
    echo "Missing Attention prototype: ${ATTENTION_PROTOTYPE}"
    exit 1
fi
if [[ ! -f "${ATTENTION_MANIFEST}" ]]; then
    echo "Missing Attention manifest: ${ATTENTION_MANIFEST}"
    echo "Run: python dataset_maker/make_Attention.py --resume"
    exit 1
fi

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG_DIR="outputs/selected_batch_logs"
BATCH_LOG="${BATCH_LOG_DIR}/run_37_40_44_selected_${BATCH_ID}.log"
mkdir -p "${BATCH_LOG_DIR}"

exec > >(tee -a "${BATCH_LOG}") 2>&1

TASK_NAMES=()
TASK_SCRIPTS=()
TASK_SEEDS=()

add_task() {
    local name="$1"
    local script="$2"
    local seed="$3"
    TASK_NAMES+=("${name}")
    TASK_SCRIPTS+=("${script}")
    TASK_SEEDS+=("${seed}")
}

for seed in "${EEGMAT_SEEDS[@]}"; do
    add_task "37Nada_eegmat8_seed${seed}" "${EEGMAT_N_SCRIPT}" "${seed}"
done
for seed in "${EEGMAT_SEEDS[@]}"; do
    add_task "37Aada_eegmat8_proto_seed${seed}" "${EEGMAT_A_SCRIPT}" "${seed}"
done
for seed in "${SIENA_SEEDS[@]}"; do
    add_task "40Aada_siena13_proto_seed${seed}" "${SIENA_A_SCRIPT}" "${seed}"
done
for seed in "${SIENA_SEEDS[@]}"; do
    add_task "40Nada_siena13_seed${seed}" "${SIENA_N_SCRIPT}" "${seed}"
done
for seed in "${ATTENTION_SEEDS[@]}"; do
    add_task "44O_attention26_full_seed${seed}" "${ATTENTION_FULL_SCRIPT}" "${seed}"
done
for seed in "${ATTENTION_SEEDS[@]}"; do
    add_task "44N_attention10_full_seed${seed}" "${ATTENTION_N_SCRIPT}" "${seed}"
done
for seed in "${ATTENTION_SEEDS[@]}"; do
    add_task "44A_attention10_proto_full_seed${seed}" "${ATTENTION_A_SCRIPT}" "${seed}"
done

total_runs="${#TASK_SCRIPTS[@]}"

run_one() {
    local task_index="$1"
    local gpu="$2"
    local display_index=$((task_index + 1))
    local name="${TASK_NAMES[task_index]}"
    local script="${TASK_SCRIPTS[task_index]}"
    local seed="${TASK_SEEDS[task_index]}"
    local port=$((29700 + task_index))
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
        echo "TASK_GPUS=auto requires nvidia-smi, but it was not found"
        exit 1
    fi
    CANDIDATE_GPUS=()
    while IFS=',' read -r raw_index; do
        gpu_index="$(echo "${raw_index}" | tr -d '[:space:]')"
        if [[ -n "${gpu_index}" ]]; then
            CANDIDATE_GPUS+=("${gpu_index}")
        fi
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    if [[ "${#CANDIDATE_GPUS[@]}" -eq 0 ]]; then
        echo "No GPU found by nvidia-smi"
        exit 1
    fi
else
    IFS=',' read -r -a CANDIDATE_GPUS <<< "${TASK_GPUS}"
fi

if [[ "${#CANDIDATE_GPUS[@]}" -lt 1 ]]; then
    echo "TASK_GPUS must contain at least one GPU index"
    exit 1
fi
for gpu in "${CANDIDATE_GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in TASK_GPUS: ${gpu}"
        exit 1
    fi
done

gpu_is_reserved() {
    local gpu="$1"
    local reserved_gpu
    for reserved_gpu in "${RUNNING_GPUS[@]:-}"; do
        if [[ "${reserved_gpu}" == "${gpu}" ]]; then
            return 0
        fi
    done
    return 1
}

find_idle_gpu() {
    local raw_index raw_memory raw_util gpu_index memory_used gpu_util candidate
    while IFS=',' read -r raw_index raw_memory raw_util; do
        gpu_index="$(echo "${raw_index}" | tr -d '[:space:]')"
        memory_used="$(echo "${raw_memory}" | tr -d '[:space:]')"
        gpu_util="$(echo "${raw_util}" | tr -d '[:space:]')"
        if [[ -z "${gpu_index}" || -z "${memory_used}" || -z "${gpu_util}" ]]; then
            continue
        fi
        for candidate in "${CANDIDATE_GPUS[@]}"; do
            if [[ "${gpu_index}" != "${candidate}" ]]; then
                continue
            fi
            if gpu_is_reserved "${gpu_index}"; then
                continue
            fi
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
    local i pid status running_jobs

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

echo "Selected batch run"
echo "Batch id: ${BATCH_ID}"
echo "Batch log: ${BATCH_LOG}"
echo "Candidate GPUs: ${CANDIDATE_GPUS[*]}"
echo "GPU idle thresholds: memory.used <= ${GPU_FREE_MEMORY_MAX_MB} MiB, utilization.gpu <= ${GPU_FREE_UTIL_MAX}%"
echo "GPU poll seconds: ${GPU_POLL_SECONDS}"
echo "Total runs: ${total_runs}"
echo "EEGMAT seeds: ${EEGMAT_SEEDS[*]}"
echo "Siena seeds: ${SIENA_SEEDS[*]}"
echo "Attention seeds: ${ATTENTION_SEEDS[*]}"
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
        echo "Candidate GPUs: ${CANDIDATE_GPUS[*]}"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
        sleep "${GPU_POLL_SECONDS}"
    fi
done

if [[ "${parallel_status}" != "0" ]]; then
    echo "At least one worker failed; inspect ${BATCH_LOG} and per-run logs"
    exit "${parallel_status}"
fi

echo
echo "Completed all ${total_runs} selected runs"
echo "Batch log: ${BATCH_LOG}"
