#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME%.sh}"
OUTPUT_SCRIPT_NAME="${OUTPUT_SCRIPT_NAME//./_}"
export OUTPUT_SCRIPT_NAME
# PhysioNet: use 32 real channels, complete to 64 with train-set CNN
# prototypes, freeze CNN, and train the Transformer plus classification head.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/34Oeegfm.finetune_physionet_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="physionet32"
export COMPLETION_SCOPE="physionet32_with_physionet64"
export CHANNEL_PROTOTYPE_PATH="${CHANNEL_PROTOTYPE_PATH:-docs/prototypes/01_physionet64_cnn_patch_embed_mean.pth}"
export POOLING_SCOPE="${POOLING_SCOPE:-high}"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-real}"
export EXP_GROUP="preexp34_physionet32_with_physionet64_prototype_motor_imagery"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29507}"

exec bash "${BASE_SCRIPT}" "$@"
