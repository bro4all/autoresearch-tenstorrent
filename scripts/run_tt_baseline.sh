#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-tt}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-tt_singlechip}"
export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-${AUTORESEARCH_TT_VISIBLE_DEVICES:-0}}"

exec python train.py "$@"
