#!/usr/bin/env bash
# Run a command inside the TT-XLA Docker container with device access.
# Usage: ./scripts/docker_run.sh <command...>
# Example: ./scripts/docker_run.sh python train.py --experiment --description baseline
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$HOME/.cache/autoresearch}"
mkdir -p "${CACHE_DIR}"

docker run --rm \
  -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}" \
  -e PJRT_DEVICE=TT \
  -e XLA_STABLEHLO_COMPILE=1 \
  -e AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-tt}" \
  -e AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-tt_singlechip}" \
  -e AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-}" \
  -e AUTORESEARCH_MAX_SEQ_LEN="${AUTORESEARCH_MAX_SEQ_LEN:-}" \
  -e AUTORESEARCH_DEPTH="${AUTORESEARCH_DEPTH:-}" \
  -e AUTORESEARCH_TOTAL_BATCH_SIZE="${AUTORESEARCH_TOTAL_BATCH_SIZE:-}" \
  -e AUTORESEARCH_DEVICE_BATCH_SIZE="${AUTORESEARCH_DEVICE_BATCH_SIZE:-}" \
  -e AUTORESEARCH_EVAL_TOKENS="${AUTORESEARCH_EVAL_TOKENS:-}" \
  -e AUTORESEARCH_LEARNING_RATE="${AUTORESEARCH_LEARNING_RATE:-}" \
  -e AUTORESEARCH_WARMUP_STEPS="${AUTORESEARCH_WARMUP_STEPS:-}" \
  -e AUTORESEARCH_BF16="${AUTORESEARCH_BF16:-}" \
  -e AUTORESEARCH_FREEZE_EMBEDDINGS="${AUTORESEARCH_FREEZE_EMBEDDINGS:-}" \
  -e AUTORESEARCH_SEED="${AUTORESEARCH_SEED:-}" \
  -e AUTORESEARCH_ENABLE_TT_COMPILE="${AUTORESEARCH_ENABLE_TT_COMPILE:-}" \
  -e AUTORESEARCH_WEIGHT_DECAY="${AUTORESEARCH_WEIGHT_DECAY:-}" \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "${WORKSPACE}":/workspace \
  -v "${CACHE_DIR}":/root/.cache/autoresearch \
  -w /workspace \
  ghcr.io/tenstorrent/tt-xla-slim:latest \
  bash -c "pip install -e . -q 2>/dev/null && $*"
