#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$ROOT_DIR/.cache-repro}"
RUN1_LOG="${ROOT_DIR}/run_repro_1.log"
RUN2_LOG="${ROOT_DIR}/run_repro_2.log"

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-tt}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-smoke}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-60}"
export AUTORESEARCH_SEED="${AUTORESEARCH_SEED:-123}"
export AUTORESEARCH_CACHE_DIR="${CACHE_DIR}"

python prepare.py --smoke --synthetic
python train.py >"${RUN1_LOG}" 2>&1
python train.py >"${RUN2_LOG}" 2>&1

python - <<'PY'
from pathlib import Path

def parse(path: Path):
    data = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data

r1 = parse(Path("run_repro_1.log"))
r2 = parse(Path("run_repro_2.log"))
v1, v2 = float(r1["val_bpb"]), float(r2["val_bpb"])
s1, s2 = float(r1["num_steps"]), float(r2["num_steps"])
print("val_bpb delta:", abs(v1 - v2))
print("step delta frac:", abs(s1 - s2) / max(s1, 1.0))
PY
