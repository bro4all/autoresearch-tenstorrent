# autoresearch-tenstorrent

This is the Tenstorrent single-device port of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch).

The idea is the same as upstream: give an AI agent a small but real LLM training setup and let it experiment autonomously on a fixed wall-clock budget. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The TT port keeps that research loop contract intact while replacing the CUDA/H100 assumptions with a correctness-first TT-XLA baseline that runs on one Tenstorrent device.

This repo is not an in-place edit of upstream. It is a fresh TT-focused implementation that stays close to upstream semantics where it matters: fixed training budget, `val_bpb` as the optimization target, `results.tsv` logging, and a minimal repo shape that a future autonomous coding loop can still use.

## Human control surface

Like upstream, this project is meant to be steered from two directions:

- the **agent** mainly edits `train.py`
- the **human** mainly edits `program.md`

That split matters. `train.py` is the search surface for model and training ideas, while `program.md` is the research org code that tells the agent how to validate, when to revert, what counts as evidence, and how cautious it should be with TT hardware. In practice you are programming both the model and the local research protocol.

## How it works

The repo is still deliberately small. The files that matter most are:

- **`prepare.py`**: fixed constants, one-time data prep, tokenizer, dataloader, evaluation. Keep this frozen during research runs.
- **`train.py`**: the single file the agent edits during autonomous experimentation. Model, optimizer, training loop, architecture, and research ideas live here.
- **`program.md`**: the baseline agent protocol for TT runs. This is where the human encodes the experiment workflow and guardrails.

The TT port adds a few support files around that core:

- **`configs.py`**: named profiles and environment overrides
- **`tt_runtime.py`**: runtime/device glue so the code never branches on CUDA directly
- **`scripts/run_tt_smoke.sh` / `scripts/run_tt_baseline.sh`**: the preferred TT entrypoints with host-side preflight and recovery

By design, training runs for a **fixed 5-minute time budget** excluding early compile/warmup steps when practical. The metric is still **`val_bpb`** (validation bits per byte) and its meaning is unchanged from upstream. This keeps experiments comparable within the same hardware/runtime path even when the agent changes architecture or batch shape.

If you are new to neural networks, upstream links to this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) for extra context.

## Quick start

**Requirements:** one Tenstorrent device, Ubuntu 22.04-class host, Python 3.12, and a documented TT-XLA runtime path. The documented default path for this repo is `pip` and shell scripts, not `uv`.

Wheel path:

```bash
cd autoresearch-tenstorrent
./scripts/bootstrap_tt.sh
source .venv-tt/bin/activate
python prepare.py
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh
```

Docker path:

```bash
export TT_VISIBLE_DEVICES=0
docker run -it --rm \
  -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES}" \
  -e PJRT_DEVICE=TT \
  -e XLA_STABLEHLO_COMPILE=1 \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/tenstorrent/tt-xla-slim:latest
```

Inside the container:

```bash
pip install -e .
./scripts/check_tt_env.sh
python prepare.py
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh
```

If the above commands work, the TT path is live and you can move into autonomous research mode.

## Autonomous research workflow

Point your coding agent at [`program.md`](/workdir/autoresearch-tenstorrent/program.md). A minimal prompt is still:

```text
Hi, have a look at program.md and let's kick off a new experiment. Let's do the setup first.
```

The protocol remains intentionally lightweight. The main difference from upstream is that the baseline training path is TT-XLA on one Tenstorrent device, and `train.py` is still the intended mutation target during experimentation.

A typical loop on this fork looks like:

1. Establish the real TT baseline on `AUTORESEARCH_PROFILE=tt_singlechip`.
2. Try a small candidate edit in `train.py`.
3. Screen it with a 60-second TT smoke run first.
4. Promote only promising candidates to the full 300-second `tt_singlechip` baseline.
5. Revert regressions immediately and keep only changes that help `val_bpb` and do not materially hurt TT throughput.
6. Keep `results.tsv` as a local run log, but keep the git-tracked file header-only.

That is the intended "autoresearch" loop here: fast TT smoke for triage, full TT baseline for evidence, and minimal code churn outside `train.py`.

## Project structure

```text
prepare.py       — constants, data prep + runtime utilities (do not modify in research loop)
train.py         — model, optimizer, training loop (agent modifies this)
program.md       — agent instructions
configs.py       — profiles and env overrides
tt_runtime.py    — TT/CPU runtime helpers
pyproject.toml   — Python dependencies
```

## Design choices

- **Single file to modify.** Autonomous experiments should mutate only `train.py`.
- **Fixed time budget.** Training is compared on the same wall-clock budget, not the same step count.
- **TT-XLA first.** The primary backend is idiomatic lazy `torch_xla` on TT, not `tt-torch`, not TT-Forge-FE as the main path, and not whole-model `torch.compile(backend="tt")`.
- **Correctness-first baseline.** FlashAttention, CUDA-only fused optimizer paths, CUDA memory metrics, and H100-specific MFU logic are removed or replaced.
- **Minimal support layer.** The port adds `configs.py`, `tt_runtime.py`, tests, and shell wrappers so the rest of the code stays backend-agnostic.

## What differs from upstream

- Primary backend is TT-XLA on Tenstorrent, not CUDA on NVIDIA.
- Baseline attention is pure PyTorch causal attention compatible with CPU and TT-XLA.
- Baseline optimizer is standard `AdamW`.
- `profile=tt_singlechip` is the default profile and is much smaller than the H100-oriented upstream defaults.
- Sliding-window attention and whole-model compile stay behind flags and are default-off.
- TT-only unavailable metrics such as `peak_vram_mb` and `mfu_percent` print `-1.0`.
- Token/value embeddings are now unfrozen by default (previously frozen for stability with the lower LR; stable at LR=0.01 with beta1=0.8).
- MLP expansion is 6x (upstream uses 4x; 6x fits in N300 DRAM and improves val_bpb).
- Adam beta1 is 0.8 (matches upstream's AdamW groups).
- LR warmdown schedule (30% of budget, floor 0.1) is applied in the training loop.

Full deviation log: [`docs/PORTING_NOTES.md`](/workdir/autoresearch-tenstorrent/docs/PORTING_NOTES.md)

## Platform support

Upstream targets a single NVIDIA GPU. This fork targets a single Tenstorrent device through TT-XLA.

Supported baseline path:

- one Wormhole-class device
- Ubuntu 22.04-class host
- Python 3.12
- TT-XLA wheel install or `ghcr.io/tenstorrent/tt-xla-slim:latest`

Recommended runtime validation:

```bash
./scripts/check_tt_env.sh
```

Equivalent direct probes:

```bash
export TT_VISIBLE_DEVICES=0
python -c "import jax; print(jax.devices('tt'))"
python - <<'PY'
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
xr.set_device_type("TT")
print(xm.xla_device())
PY
```

Like upstream, this code is intentionally not trying to be every platform at once. The baseline path here is one TT device. CPU exists for smoke tests and debugging only.

## Tuning smaller TT runs

The upstream README recently added guidance for much smaller machines. The TT fork already bakes a lot of that thinking into `profile=smoke` and `profile=tt_singlechip`, but the same knobs still matter here:

1. Lower `AUTORESEARCH_MAX_SEQ_LEN` first if you need an easier compiler/runtime shape.
2. Lower `AUTORESEARCH_EVAL_TOKENS` if evaluation overhead dominates iteration time.
3. Lower `AUTORESEARCH_DEPTH` to reduce model size quickly while preserving the same training loop and metric.
4. Keep `AUTORESEARCH_WINDOW_PATTERN=L` and `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0` unless you are intentionally validating the alternate attention path.
5. Sweep `AUTORESEARCH_DEVICE_BATCH_SIZE` and `AUTORESEARCH_TOTAL_BATCH_SIZE` together. On the tested N300 path, larger device microbatches were the main throughput win.

If you are forking this repo to another TT stack or a much smaller shape, start from `profile=smoke`, keep the TT-specific guardrails on, and only then scale features back up.

The right bring-up order is:

1. synthetic or offline-friendly smoke data first
2. real prepared data once the runtime path is stable
3. only then consider changing tokenizer size, data complexity, or other prepare-path assumptions

That last category is a separate porting exercise, not part of the default autonomous loop in this repo. During normal research runs, `prepare.py` stays frozen and only `train.py` should move.

## Preparing data

Full upstream-compatible data and tokenizer prep:

```bash
python prepare.py
```

Tiny offline smoke cache:

```bash
AUTORESEARCH_PROFILE=smoke AUTORESEARCH_BACKEND=cpu python prepare.py --smoke --synthetic
```

## Common commands

CPU smoke:

```bash
./scripts/run_cpu_smoke.sh
```

TT environment check:

```bash
TT_VISIBLE_DEVICES=0 ./scripts/check_tt_env.sh
```

60-second TT smoke:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=smoke AUTORESEARCH_TIME_BUDGET=60 ./scripts/run_tt_smoke.sh
```

300-second TT baseline:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh
```

## Comparison contract

What counts as a real result on this fork:

- Compare runs only on the same hardware path and same profile.
- TT runs on the connected N300 are the benchmark evidence.
- CPU runs are for cheap smoke tests and debugging only.
- `tokens_per_sec_avg` here is training throughput for a small GPT-style causal LM in this repo, not inference throughput for a pretrained LLM.
- `val_bpb` semantics stay fixed. If that definition changes, the comparison is no longer valid.

## Measured baselines

Measured on the connected N300 using `ghcr.io/tenstorrent/tt-xla-slim:latest`.

60-second `smoke` reference:

```text
backend: tt
tt_device: xla:0
init_val_bpb: 1.822148
val_bpb: 1.433131
training_seconds: 60.067284
total_seconds: 160.884184
peak_vram_mb: -1.000000
mfu_percent: -1.000000
total_tokens_M: 0.036608
num_steps: 143
num_params_M: 0.123874
depth: 1
tokens_per_sec_avg: 609.449898
```

300-second `tt_singlechip` reference:

```text
backend: tt
tt_device: xla:0
init_val_bpb: 3.274888
val_bpb: 1.983337
training_seconds: 301.952808
total_seconds: 479.539624
peak_vram_mb: -1.000000
mfu_percent: -1.000000
total_tokens_M: 2.621440
num_steps: 160
num_params_M: 3.670084
depth: 2
tokens_per_sec_avg: 8681.621543
```

On the same device and profile, future changes should not reduce `tokens_per_sec_avg` by more than 20% unless they improve `val_bpb` materially.

## Current best

This repo is intentionally a tiny single-device TT research rig, not a production inference stack. The current verified best baseline on the connected N300 is the 300-second `tt_singlechip` run above: `val_bpb 1.983337`, `tokens_per_sec_avg 8681.621543`, `num_steps 160`.

Key optimizations over the original baseline (`val_bpb 2.421`):
- Learning rate increased from 6e-4 to 1e-2 (16.7x higher)
- Adam beta1 reduced from 0.9 to 0.8 (matches upstream)
- MLP expansion increased from 4x to 6x
- Token/value embeddings unfrozen (were frozen for stability)
- 30% LR warmdown schedule with 0.1 floor added

That is the bar future work should beat while preserving `val_bpb` semantics and keeping TT-XLA as the primary backend.

## Profiles

`profile=upstreamish`
- Keeps the upstream-scale semantics as closely as practical.
- Mainly for semantic comparison, not for the default TT path.

`profile=tt_singlechip`
- Default.
- Current verified N300-safe geometry: `max_seq_len=256`, `depth=2`, `total_batch_size=16384`, `device_batch_size=64`, `eval_tokens=262144`, `bf16=0`, `lr=0.01`, `freeze_embeddings=0`.
- Keeps the 300-second budget and the same `val_bpb` definition.

`profile=smoke`
- Extra-small synthetic-friendly path for tests and quick TT validation.

## Environment overrides

Supported env overrides include:

- `AUTORESEARCH_BACKEND`
- `AUTORESEARCH_PROFILE`
- `AUTORESEARCH_TIME_BUDGET`
- `AUTORESEARCH_MAX_SEQ_LEN`
- `AUTORESEARCH_EVAL_TOKENS`
- `AUTORESEARCH_DEPTH`
- `AUTORESEARCH_TOTAL_BATCH_SIZE`
- `AUTORESEARCH_DEVICE_BATCH_SIZE`
- `AUTORESEARCH_WINDOW_PATTERN`
- `AUTORESEARCH_ENABLE_SLIDING_WINDOW`
- `AUTORESEARCH_ENABLE_TT_COMPILE`
- `AUTORESEARCH_SEED`
- `AUTORESEARCH_BF16`
- `AUTORESEARCH_FREEZE_EMBEDDINGS`
- `AUTORESEARCH_CACHE_DIR`

## Research loop

The research loop is still centered on `results.tsv`:

```text
commit	val_bpb	memory_gb	status	description
```

Treat the checked-in `results.tsv` as a header/template file. Log local runs there while working, but do not commit evolving experiment history back to git.

See [`program.md`](/workdir/autoresearch-tenstorrent/program.md) for the TT-specific protocol.

## Manual upstream sync

Keep this repo in sync with upstream by hand:

```bash
./scripts/check_upstream_sync.sh
```

This repo does not share a normal merge history with upstream, so the sync policy is:

1. fetch `upstream/master`
2. review the upstream commits and key-file diffs
3. selectively port backend-agnostic fixes and docs/process changes
4. re-run TT validation before pushing

Detailed process: [`docs/UPSTREAM_SYNC.md`](/workdir/autoresearch-tenstorrent/docs/UPSTREAM_SYNC.md)

## Troubleshooting

If a TT run fails with unsupported ops or lazy-graph issues:

- re-run the smallest repro in [`debug/repro_attention.py`](/workdir/autoresearch-tenstorrent/debug/repro_attention.py)
- force CPU with `AUTORESEARCH_BACKEND=cpu`
- keep `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0`
- keep `AUTORESEARCH_ENABLE_TT_COMPILE=0`
- keep `AUTORESEARCH_BF16=0` on the current winning N300 path; BF16 currently crashes in the TT MLIR compiler on this stack

If a TT run fails before model code with dispatch/fabric errors:

- use [`scripts/check_tt_env.sh`](/workdir/autoresearch-tenstorrent/scripts/check_tt_env.sh), [`scripts/run_tt_smoke.sh`](/workdir/autoresearch-tenstorrent/scripts/run_tt_smoke.sh), or [`scripts/run_tt_baseline.sh`](/workdir/autoresearch-tenstorrent/scripts/run_tt_baseline.sh)
- set `TT_VISIBLE_DEVICES=0` before importing `jax` or `torch_xla`
- let the shell wrappers own host-side reset/preflight on flaky N300 hosts

Known limitations: [`docs/KNOWN_LIMITATIONS.md`](/workdir/autoresearch-tenstorrent/docs/KNOWN_LIMITATIONS.md)

## License

MIT. Upstream-derived logic from `karpathy/autoresearch` is retained under the same license terms.
