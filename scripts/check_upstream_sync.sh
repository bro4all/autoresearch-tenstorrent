#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

UPSTREAM_REMOTE="${AUTORESEARCH_UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${AUTORESEARCH_UPSTREAM_BRANCH:-master}"
UPSTREAM_REF="${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"
MAX_COMMITS="${AUTORESEARCH_UPSTREAM_MAX_COMMITS:-20}"
KEY_FILES=(README.md prepare.py train.py program.md pyproject.toml)

if ! git remote get-url "${UPSTREAM_REMOTE}" >/dev/null 2>&1; then
  echo "Missing git remote '${UPSTREAM_REMOTE}'." >&2
  exit 1
fi

git fetch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}" >/dev/null

HEAD_SHA="$(git rev-parse HEAD)"
UPSTREAM_SHA="$(git rev-parse "${UPSTREAM_REF}")"
UPSTREAM_SUBJECT="$(git log -1 --format=%s "${UPSTREAM_REF}")"
UPSTREAM_DATE="$(git log -1 --format=%cs "${UPSTREAM_REF}")"

read -r LEFT_COUNT RIGHT_COUNT < <(git rev-list --left-right --count "${UPSTREAM_REF}...HEAD")
MERGE_BASE="$(git merge-base "${UPSTREAM_REF}" HEAD || true)"

echo "repo_head: ${HEAD_SHA}"
echo "upstream_ref: ${UPSTREAM_REF}"
echo "upstream_head: ${UPSTREAM_SHA}"
echo "upstream_date: ${UPSTREAM_DATE}"
echo "upstream_subject: ${UPSTREAM_SUBJECT}"
echo "raw_upstream_only_commits: ${LEFT_COUNT}"
echo "raw_local_only_commits: ${RIGHT_COUNT}"
if [[ -n "${MERGE_BASE}" ]]; then
  echo "merge_base: ${MERGE_BASE}"
else
  echo "merge_base: none"
  echo "note: local TT port history does not share a normal merge-base with upstream; use selective porting, not merge/rebase."
fi

echo
echo "recent_upstream_commits:"
git log --oneline --max-count="${MAX_COMMITS}" "${UPSTREAM_REF}"

echo
echo "key_file_diff_stats:"
for path in "${KEY_FILES[@]}"; do
  printf -- "--- %s\n" "${path}"
  git diff --stat "${UPSTREAM_REF}" -- "${path}" || true
done

echo
echo "recommended_validation:"
if git diff --quiet "${UPSTREAM_REF}" -- README.md program.md pyproject.toml; then
  echo "- docs-only delta: no TT validation required beyond manual review."
else
  echo "- docs/config delta: run ./scripts/run_cpu_smoke.sh"
fi
if ! git diff --quiet "${UPSTREAM_REF}" -- prepare.py; then
  echo "- prepare.py delta: run python prepare.py --smoke --synthetic"
  echo "- prepare.py delta: run ./scripts/run_cpu_smoke.sh"
  echo "- prepare.py delta: run TT smoke after porting: AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=smoke AUTORESEARCH_TIME_BUDGET=60 ./scripts/run_tt_smoke.sh"
fi
if ! git diff --quiet "${UPSTREAM_REF}" -- train.py; then
  echo "- train.py delta: run ./scripts/run_cpu_smoke.sh"
  echo "- train.py delta: run TT correctness: pytest -q tests/test_tt_correctness.py::test_tiny_cpu_vs_tt -s"
  echo "- train.py delta: run TT smoke: pytest -q tests/test_tt_train_smoke.py::test_60s_smoke_run -s"
  echo "- train.py delta: run TT baseline: pytest -q tests/test_tt_train_smoke.py::test_300s_baseline_run -s"
fi

echo
echo "selective_port_workflow:"
echo "1. Review upstream commits above and diff the key files manually."
echo "2. Port only backend-agnostic fixes or docs/process changes that still make sense on TT."
echo "3. Do not merge upstream history directly into this branch."
echo "4. Re-run the recommended validation commands before pushing."
