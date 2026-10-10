#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${RUN_DIR}/plot_stage1_fullchannel.sh" --split train "$@"
