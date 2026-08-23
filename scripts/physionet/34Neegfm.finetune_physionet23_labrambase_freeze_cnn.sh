#!/usr/bin/env bash
set -euo pipefail
# PhysioNet 23-channel CNN-frozen fine-tuning. Reuse the audited 64-channel
# launcher while overriding the subset, output namespace, and rendezvous port.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/34Oeegfm.finetune_physionet_labrambase_freeze_cnn.sh"


export CHANNEL_SUBSET="physionet23"
export CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
export EXP_GROUP="preexp34_physionet23_motor_imagery"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29503}"

exec bash "${BASE_SCRIPT}" "$@"
