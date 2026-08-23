#!/usr/bin/env bash
set -euo pipefail

# PreExp37 EEGMAT reduced-channel baseline: use an 8-channel subset while
# reusing the audited EEGMAT mean-pooling launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh"

export CHANNEL_SUBSET="eegmat8"
export EXP_GROUP="${EXP_GROUP:-preexp37_eegmat8_mean_pool_cross_subject}"
export RUN_PREFIX_OVERRIDE="$(basename "${BASH_SOURCE[0]}" .sh)"
export MASTER_PORT="${MASTER_PORT:-29518}"

exec bash "${BASE_SCRIPT}" "$@"
