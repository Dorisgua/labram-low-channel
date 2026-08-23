#!/usr/bin/env bash
set -euo pipefail

# PreExp37 EEGMAT AdaBrain-style prototype completion:
# use 8 real EEGMAT channels, complete to EEGMAT-19 with train-set CNN
# patch_embed prototypes, use all-token constrained head, freeze CNN.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Oada.finetune_eegmat19_labrambase_full_finetune.sh"

export CHANNEL_SUBSET="${CHANNEL_SUBSET:-eegmat8}"
export COMPLETION_SCOPE="${COMPLETION_SCOPE:-eegmat8_with_eegmat19}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_eegmat19_cnn_patch_embed_mean.pth}"
export FREEZE_CNN="1"
export LR="${LR:-5e-4}"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat8_with_eegmat19_adabrain_freeze_cnn_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29528}"

exec bash "${BASE_SCRIPT}" "$@"
