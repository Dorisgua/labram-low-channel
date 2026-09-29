#!/usr/bin/env bash
set -euo pipefail

# Train LaBraM D classifier on the same CBraMod subject split.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the CBraMod Stage 1 checkpoint-best.pth}"
export DATASET="bciiv2a_cbramod"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/data_splits/bciiv2a_cbramod_cross_subject_json}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_bciiv2a22_cbramod_train_cnn_patch_embed_mean.pth}"
export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-bciiv2a_cbramod_D_stage2}"
exec bash "${SCRIPT_DIR}/stage2.sh" "$@"
