#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/tt_common.sh"

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-tt}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-tt_singlechip}"
export AUTORESEARCH_TT_RESET_BEFORE_INIT="${AUTORESEARCH_TT_RESET_BEFORE_INIT:-0}"
tt_set_visible_devices
tt_host_preflight
tt_maybe_host_reset

tt_run_with_recovery python train.py "$@"
