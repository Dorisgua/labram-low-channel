#!/usr/bin/env bash
set -euo pipefail

# LaBraM D with CBraMod's subject-disjoint BCI-IV-2A split.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export DATASET="bciiv2a_cbramod"
export DATA_PATH="${DATA_PATH:-${REPO_DIR}/data_splits/bciiv2a_cbramod_cross_subject_json}"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-${REPO_DIR}/docs/prototypes/01_bciiv2a22_cbramod_train_cnn_patch_embed_mean.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/bciiv2a_cbramod/D_stage1/seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)_$$}"
exec bash "${SCRIPT_DIR}/stage1.sh" "$@"
