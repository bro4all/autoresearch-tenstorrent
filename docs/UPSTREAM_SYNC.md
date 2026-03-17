# Upstream Sync

This repo is intentionally attached to `karpathy/autoresearch` on GitHub for discoverability, but the TT port was created as its own history. Treat upstream sync as selective porting, not a normal branch merge.

## Why Normal Merge Is Wrong Here

- `origin/main` is not a normal descendant of `upstream/master`.
- The TT port rewrote the runtime, training loop, and repo structure around TT-XLA.
- A direct `git merge upstream/master` would mix unrelated histories and create low-signal conflicts.

Use upstream as a source of ideas, bug fixes, and protocol changes. Port them intentionally into the TT codebase.

## Repo Workflow

Start every manual upstream review with:

```bash
./scripts/check_upstream_sync.sh
```

That script:

- fetches `upstream/master`
- prints the raw ahead/behind counts
- reports whether a normal merge-base exists
- lists recent upstream commits
- shows diff stats for the upstream key files:
  - `README.md`
  - `prepare.py`
  - `train.py`
  - `program.md`
  - `pyproject.toml`
- prints the TT validation commands you should run after porting changes

## Porting Rules

Port upstream changes by category:

- `README.md` / `program.md`
  - Carry over protocol or workflow changes that still make sense for TT.
  - Keep TT-specific hardware/runtime instructions intact.
- `prepare.py`
  - Prefer upstream correctness fixes.
  - Preserve upstream shard/tokenizer/data layout compatibility when practical.
  - Re-verify `val_bpb` semantics after every prepare-path change.
- `train.py`
  - Port backend-agnostic bug fixes.
  - Do not import CUDA-only fast paths, FlashAttention assumptions, fused optimizer code, or H100-specific metrics directly.
  - Translate useful ideas into TT-XLA-compatible PyTorch if needed.
- `pyproject.toml`
  - Keep pip/shell bootstrap as the documented default path.
  - Do not switch the documented install path to `uv`.

## Validation Matrix

After porting upstream changes, validate based on what changed.

Minimum checks:

```bash
./scripts/run_cpu_smoke.sh
```

If `prepare.py` changed:

```bash
python prepare.py --smoke --synthetic
./scripts/run_cpu_smoke.sh
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=smoke AUTORESEARCH_TIME_BUDGET=60 ./scripts/run_tt_smoke.sh
```

If `train.py` changed:

```bash
./scripts/run_cpu_smoke.sh
pytest -q tests/test_tt_correctness.py::test_tiny_cpu_vs_tt -s
pytest -q tests/test_tt_train_smoke.py::test_60s_smoke_run -s
pytest -q tests/test_tt_train_smoke.py::test_300s_baseline_run -s
```

If the TT board is flaky, use the shell wrappers and environment probe first:

```bash
./scripts/check_tt_env.sh
```

## Sync Cadence

Recommended cadence:

1. Fetch/report upstream before any meaningful TT tuning pass.
2. Port obvious upstream bug fixes immediately.
3. Port docs/process changes in batches.
4. Re-run the TT smoke and TT baseline checks after any `train.py` or `prepare.py` sync.

## What To Track

After each upstream review, note:

- upstream commit reviewed
- which changes were ported
- which were intentionally skipped
- TT validation commands run
- measured TT smoke/baseline results if they changed

Update [`docs/PORTING_NOTES.md`](PORTING_NOTES.md) or the README if the TT defaults or known deviations change.
