# autoresearch-tenstorrent

This is the Tenstorrent single-device port of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch).

The idea is the same as upstream: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The TT port keeps that research loop contract intact while replacing the CUDA/H100 assumptions with a TT-XLA baseline that runs on one Tenstorrent device. The core idea is that you're not touching the Python files directly — instead, you are programming the `program.md` that provides context to the AI agents. The agent edits `train.py`, the human edits `program.md`.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified during research.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

The TT port adds a few support files: `configs.py` (named profiles and env overrides), `tt_runtime.py` (device glue so the code never branches on CUDA directly), and shell wrappers in `scripts/`.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, upstream links to this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) for extra context.

## Quick start

**Requirements:** one Tenstorrent Wormhole-class device (tested on N300), Ubuntu 22.04-class host, Python 3.12, and the TT-XLA runtime.

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
python prepare.py
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh
```

If the above commands work, the TT path is live and you can move into autonomous research mode.

## Running the agent

Simply spin up Claude/Codex or whatever you want in this repo, then prompt something like:

```
Hi, have a look at program.md and let's kick off a new experiment. Let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill". The main difference from upstream is that the baseline training path is TT-XLA on one Tenstorrent device.

## Project structure

```text
prepare.py       — constants, data prep + runtime utilities (do not modify)
train.py         — model, optimizer, training loop (agent modifies this)
program.md       — agent instructions
configs.py       — profiles and env overrides
tt_runtime.py    — TT/CPU runtime helpers
pyproject.toml   — Python dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of what the agent changes. This makes experiments directly comparable and means autoresearch will find the most optimal model for TT hardware in that budget.
- **TT-XLA first.** The primary backend is idiomatic lazy `torch_xla` on TT, not `tt-torch`, not TT-Forge-FE, and not `torch.compile(backend="tt")`.
- **Self-contained.** No distributed training, no complex configs. One device, one file, one metric.

## What differs from upstream

- Primary backend is TT-XLA on Tenstorrent, not CUDA on NVIDIA.
- Baseline attention is pure PyTorch causal attention (no FlashAttention 3).
- Baseline optimizer is `AdamW` (no Muon — requires `torch.compile` which isn't TT-stable yet).
- `profile=tt_singlechip` is much smaller than the H100-oriented upstream defaults.
- MLP expansion is 6x (upstream uses 4x; 6x fits in N300 DRAM and improves val_bpb).
- Logit softcap is disabled (saves compute, improves val_bpb at this scale).
- LR warmdown schedule (30% of budget, floor 0.1) is applied in the training loop.
- Metrics like `peak_vram_mb` and `mfu_percent` print `-1.0` (not available on TT).

## Platform support

This code currently requires a single Tenstorrent Wormhole-class device. Like upstream, this code is intentionally not trying to be every platform at once. CPU exists for smoke tests and debugging only.

If you're going to try running on a smaller TT configuration, here are the knobs to tune (same spirit as upstream's guidance for smaller machines):

1. Lower `AUTORESEARCH_MAX_SEQ_LEN` first if you need an easier compiler/runtime shape.
2. Lower `AUTORESEARCH_EVAL_TOKENS` if evaluation overhead dominates iteration time.
3. Lower `AUTORESEARCH_DEPTH` to reduce model size quickly while preserving the same training loop and metric.
4. Keep `AUTORESEARCH_WINDOW_PATTERN=L` and `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0`.
5. Sweep `AUTORESEARCH_DEVICE_BATCH_SIZE` and `AUTORESEARCH_TOTAL_BATCH_SIZE` together.

Start from `profile=smoke`, keep the TT-specific guardrails on, and only then scale features back up.

## Measured baselines

Measured on the connected N300 using `ghcr.io/tenstorrent/tt-xla-slim:latest`.

300-second `tt_singlechip` (current best):

```text
backend: tt
tt_device: xla:0
init_val_bpb: 3.274888
val_bpb: 1.968179
training_seconds: 300.858213
total_seconds: 475.093116
peak_vram_mb: -1.000000
mfu_percent: -1.000000
total_tokens_M: 2.752512
num_steps: 168
num_params_M: 3.670084
depth: 2
tokens_per_sec_avg: 9148.867730
```

Key optimizations over the original baseline (`val_bpb 2.421`):
- Learning rate: 6e-4 → 1e-2 (largest single lever)
- Adam beta1: 0.9 → 0.8 (matches upstream)
- MLP expansion: 4x → 6x
- Embeddings unfrozen, weight decay removed, logit softcap disabled
- 30% LR warmdown with 0.1 floor

On the same device and profile, future changes should not reduce `tokens_per_sec_avg` by more than 20% unless they improve `val_bpb` materially.

## Known TT-XLA limitations

These features are blocked by the current TT-XLA stack, not by model code:

- **bf16**: crashes in TT MLIR compiler
- **torch.compile(backend="tt")**: runtime error in dynamo bridge
- **Multi-device (2 N300 chips)**: PJRT cross-device buffer error
- **Gradient clipping**: TTNN reshape error

When TT-XLA fixes land, these are the highest-impact optimizations to enable.

## Troubleshooting

If a TT run fails with unsupported ops or lazy-graph issues:

- force CPU with `AUTORESEARCH_BACKEND=cpu`
- keep `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0`
- keep `AUTORESEARCH_ENABLE_TT_COMPILE=0`
- keep `AUTORESEARCH_BF16=0`

If a TT run fails with dispatch/fabric errors:

- run `./scripts/check_tt_env.sh`
- set `TT_VISIBLE_DEVICES=0` before importing `torch_xla`
- use `tt-smi --reset 0` if the device is stuck after a crash

## License

MIT. Upstream-derived logic from `karpathy/autoresearch` is retained under the same license terms.
