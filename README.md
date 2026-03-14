# autoresearch-tenstorrent

This is the Tenstorrent single-device port of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch).

The idea is the same as upstream: give an AI agent a small but real LLM training setup and let it experiment autonomously on a fixed wall-clock budget. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The TT port keeps that research loop contract intact while replacing the CUDA/H100 assumptions with a correctness-first TT-XLA baseline that runs on one Tenstorrent device.

This repo is not an in-place edit of upstream. It is a fresh TT-focused implementation that stays close to upstream semantics where it matters: fixed training budget, `val_bpb` as the optimization target, `results.tsv` logging, and a minimal repo shape that a future autonomous coding loop can still use.

## How it works

The repo is still deliberately small. The files that matter most are:

- **`prepare.py`**: fixed constants, one-time data prep, tokenizer, dataloader, evaluation. Keep this frozen during research runs.
- **`train.py`**: the single file the agent edits during autonomous experimentation. Model, optimizer, training loop, architecture, and research ideas live here.
- **`program.md`**: the baseline agent protocol for TT runs.

The TT port adds a few support files around that core:

- **`configs.py`**: named profiles and environment overrides
- **`tt_runtime.py`**: runtime/device glue so the code never branches on CUDA directly
- **`scripts/run_tt_smoke.sh` / `scripts/run_tt_baseline.sh`**: the preferred TT entrypoints with host-side preflight and recovery

By design, training runs for a **fixed 5-minute time budget** excluding early compile/warmup steps when practical. The metric is still **`val_bpb`** (validation bits per byte) and its meaning is unchanged from upstream.

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

## Running the agent

Point your coding agent at [`program.md`](/workdir/autoresearch-tenstorrent/program.md). A minimal prompt is still:

```text
Hi, have a look at program.md and let's kick off a new experiment. Let's do the setup first.
```

The protocol remains intentionally lightweight. The main difference from upstream is that the baseline training path is TT-XLA on one Tenstorrent device, and `train.py` is still the intended mutation target during experimentation.

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
- TT-friendly profiles currently freeze token/value embeddings by default because the tested N300 + TT-XLA stack still becomes non-finite on the real baseline geometry when those embeddings are trained normally.

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

## Measured baselines

Measured on the connected N300 using `ghcr.io/tenstorrent/tt-xla-slim:latest`.

60-second `smoke` reference:

```text
backend: tt
tt_device: xla:0
init_val_bpb: 1.822148
val_bpb: 1.449490
training_seconds: 60.062330
total_seconds: 175.809362
peak_vram_mb: -1.000000
mfu_percent: -1.000000
total_tokens_M: 0.035072
num_steps: 137
num_params_M: 0.123874
depth: 1
tokens_per_sec_avg: 583.926726
```

300-second `tt_singlechip` reference:

```text
backend: tt
tt_device: xla:0
init_val_bpb: 3.274981
val_bpb: 2.486103
training_seconds: 302.168453
total_seconds: 461.749169
peak_vram_mb: -1.000000
mfu_percent: -1.000000
total_tokens_M: 2.310144
num_steps: 141
num_params_M: 3.539012
depth: 2
tokens_per_sec_avg: 7645.219007
```

On the same device and profile, future changes should not reduce `tokens_per_sec_avg` by more than 20% unless they improve `val_bpb` materially.

## Profiles

`profile=upstreamish`
- Keeps the upstream-scale semantics as closely as practical.
- Mainly for semantic comparison, not for the default TT path.

`profile=tt_singlechip`
- Default.
- Current verified N300-safe geometry: `max_seq_len=256`, `depth=2`, `total_batch_size=16384`, `device_batch_size=64`, `eval_tokens=262144`, `bf16=0`.
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
