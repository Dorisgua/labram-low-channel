#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/outputs/full_prototype_d/full_prototype_d_seed0_20260818_135346"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/checkpoint-last.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation/diagnostics_token_pooled}"
source "$(dirname "${BASH_SOURCE[0]}")/run_token_vs_pooled_probe.sh" "$@"
