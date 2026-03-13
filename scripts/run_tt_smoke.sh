#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-tt}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-smoke}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-60}"
export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-${AUTORESEARCH_TT_VISIBLE_DEVICES:-0}}"

exec python train.py "$@"
