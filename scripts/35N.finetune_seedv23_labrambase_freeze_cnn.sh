#!/usr/bin/env bash
set -euo pipefail
# SEED-V: 23 real channels, no prototype completion, freeze CNN.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHANNEL_SUBSET="seedv23"
export COMPLETION_SCOPE="none"
export POOLING_SCOPE="low"
export EXP_GROUP="preexp35_seedv23_freeze_cnn"
export MASTER_PORT="${MASTER_PORT:-29513}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-$(basename "${BASH_SOURCE[0]}" .sh)}"
exec bash "${SCRIPT_DIR}/35O.finetune_seedv62_labrambase_freeze_cnn.sh"
