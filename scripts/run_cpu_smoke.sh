#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-cpu}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-smoke}"
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$ROOT_DIR/.cache-smoke}"

python prepare.py --smoke --synthetic
pytest -q tests/test_cpu_smoke.py
