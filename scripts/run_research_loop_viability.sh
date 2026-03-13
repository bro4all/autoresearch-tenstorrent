#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BACKUP="$(mktemp)"
cp train.py "${BACKUP}"
trap 'cp "${BACKUP}" train.py; rm -f "${BACKUP}"' EXIT

export AUTORESEARCH_BACKEND="${AUTORESEARCH_BACKEND:-cpu}"
export AUTORESEARCH_PROFILE="${AUTORESEARCH_PROFILE:-smoke}"
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$ROOT_DIR/.cache-viability}"
export AUTORESEARCH_TIME_BUDGET="${AUTORESEARCH_TIME_BUDGET:-10}"

rm -f results.tsv
printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results.tsv
python prepare.py --smoke --synthetic
python train.py --experiment --description baseline >/tmp/autoresearch_baseline.log 2>&1

python - <<'PY'
from pathlib import Path
path = Path("train.py")
text = path.read_text()
path.write_text(text.replace("LOGIT_SOFTCAP = 15.0", "LOGIT_SOFTCAP = 12.0"))
PY
python train.py --experiment --description softcap12 >/tmp/autoresearch_variant1.log 2>&1 || true

cp "${BACKUP}" train.py
python - <<'PY'
from pathlib import Path
path = Path("train.py")
text = path.read_text()
path.write_text(text.replace("DEFAULT_MLP_EXPANSION = 4", "DEFAULT_MLP_EXPANSION = 3"))
PY
python train.py --experiment --description mlp3 >/tmp/autoresearch_variant2.log 2>&1 || true

cat results.tsv
