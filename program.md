# autoresearch-tenstorrent

This repo is the Tenstorrent single-device port of `karpathy/autoresearch`.

## Scope

During autonomous experimentation:

- Only mutate `train.py`.
- Upstream sync is a separate manual maintenance workflow. Use [`scripts/check_upstream_sync.sh`](/workdir/autoresearch-tenstorrent/scripts/check_upstream_sync.sh) and [`docs/UPSTREAM_SYNC.md`](/workdir/autoresearch-tenstorrent/docs/UPSTREAM_SYNC.md) outside the experiment loop.
- Keep `prepare.py`, `configs.py`, `tt_runtime.py`, tests, scripts, and docs frozen.
- Do not redefine `val_bpb`.
- Do not change the fixed wall-clock budget policy.

## Setup

Before the first research run:

1. Read `README.md` for the current TT runtime, profile, and baseline context.
2. Verify the TT runtime path with `./scripts/check_tt_env.sh`.
3. Verify prepared data exists under `~/.cache/autoresearch` or your `AUTORESEARCH_CACHE_DIR`.
4. Initialize `results.tsv` with:

```text
commit	val_bpb	memory_gb	status	description
```

5. Establish a baseline first. Do not mutate `train.py` before the first baseline run.
6. Keep `results.tsv` as a local experiment log. Do not commit ongoing run history back to git during the search loop.

## Baseline Command

Run the default one-device TT baseline with:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh --experiment --description baseline
```

Use `AUTORESEARCH_PROFILE=smoke` and `AUTORESEARCH_TIME_BUDGET=60` for faster TT smoke validation.

## Logging Results

Append every run to `results.tsv` as tab-separated text:

```text
commit	val_bpb	memory_gb	status	description
```

Guidelines:

- `commit`: short git hash
- `val_bpb`: final validation bpb
- `memory_gb`: `peak_vram_mb / 1024` if honestly measured, otherwise `-1.0`
- `status`: `keep`, `discard`, or `crash`
- `description`: one short sentence

## Comparison Rule

- Lower `val_bpb` is better.
- Compare runs only on the same profile and same hardware path.
- Future changes should not degrade `tokens_per_sec_avg` by more than 20% on the same device/profile unless they improve `val_bpb` materially.

## Experiment Loop

1. Read the current `results.tsv`.
2. Modify only `train.py`.
3. Commit the change.
4. Run:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh --experiment --description "<idea>"
```

5. Parse the summary block from the log.
6. If the summary is missing, read the Python traceback from the log before changing code. A minimal triage command is:

```bash
tail -n 50 run.log
```

7. If `val_bpb` improved, keep the commit.
8. If `val_bpb` regressed or the run crashed, revert only the candidate change in `train.py`.

## Hung Runs

If a TT run stops making progress:

- First inspect the current log.
- If it exceeds roughly 2x its intended wall-clock budget, kill it.
- Example:

```bash
pkill -f "run_tt_baseline.sh|python train.py"
```

Treat a hung run as `crash` in `results.tsv`.

## TT-Specific Guardrails

- Baseline path is lazy `torch_xla` execution on TT.
- Keep `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0` unless the feature has already passed TT correctness and smoke tests.
- Keep `AUTORESEARCH_ENABLE_TT_COMPILE=0` unless compile mode has already passed TT correctness and smoke tests.
- Prefer simplifying unsupported ops over changing frameworks.

## Controlled Viability Check

The repo includes a scripted three-run viability check:

```bash
./scripts/run_research_loop_viability.sh
```

It runs:

1. Baseline
2. A trivial `train.py` variant
3. A second trivial `train.py` variant

All three runs append to `results.tsv`, making the repo obviously usable by a future autonomous search loop.
