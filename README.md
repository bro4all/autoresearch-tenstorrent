# autoresearch-tenstorrent

`autoresearch-tenstorrent` is a fresh repo that ports [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) to a single Tenstorrent device. The baseline keeps the upstream research loop contract intact where it matters: fixed wall-clock training budget, `val_bpb` as the optimization target, `results.tsv` logging, and a repo shape that remains usable by an autonomous coding loop later.

This repo does **not** edit the upstream repo in place. It is a new implementation tuned for TT-XLA lazy execution on Tenstorrent hardware.

## What Differs From Upstream

- Primary backend is `torch_xla` on TT via TT-XLA, not CUDA.
- FlashAttention 3 is replaced by pure PyTorch causal attention that is compatible with CPU and TT-XLA.
- The optimizer baseline is plain `AdamW`; CUDA-only fused optimizer code is removed.
- Runtime setup is centralized in [`tt_runtime.py`](/workdir/autoresearch-tenstorrent/tt_runtime.py).
- Tunables move into [`configs.py`](/workdir/autoresearch-tenstorrent/configs.py) with named profiles and env overrides.
- Two profiles are built in:
  - `upstreamish`: close to the original defaults and semantics.
  - `tt_singlechip`: default, smaller and TT-friendly so one device can complete a real baseline run.
- Sliding-window attention and whole-model `torch.compile(backend="tt")` are optional and default-off.
- TT-only unavailable metrics such as peak VRAM and MFU print `-1.0` instead of fake values.

Full deviation log: [`docs/PORTING_NOTES.md`](/workdir/autoresearch-tenstorrent/docs/PORTING_NOTES.md)

## Supported Hardware Path

- Primary target: Ubuntu 22.04
- Primary runtime target: Python 3.12 with the TT-XLA wheel path
- Primary hardware target: one Tenstorrent device exposed through `/dev/tenstorrent`
- Baseline backend: TT-XLA lazy execution via `torch_xla`

## Install: TT-XLA Wheel Path

These steps follow the public TT-XLA getting-started flow for wheel installs and then add this repo on top.

```bash
cd autoresearch-tenstorrent
./scripts/bootstrap_tt.sh
source .venv-tt/bin/activate
```

What `bootstrap_tt.sh` does:

```bash
python3.12 -m venv .venv-tt
source .venv-tt/bin/activate
pip install --upgrade pip setuptools wheel
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install -e .
```

## Install: TT-XLA Docker Path

The official TT-XLA docs also publish a Docker path. Start the container:

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
```

## Required Environment Checks

This repo supports the two validation commands requested in the project brief. The shell wrapper runs them as two separate processes:

```bash
./scripts/check_tt_env.sh
```

Equivalent direct commands:

```bash
export TT_VISIBLE_DEVICES=0
python -c "import jax; print(jax.devices('tt'))"
python - <<'PY'
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
xr.set_device_type('TT')
print(xm.xla_device())
PY
```

## Data Preparation

Full upstream-compatible data/tokenizer prep:

```bash
python prepare.py
```

Fast synthetic smoke cache for tests:

```bash
AUTORESEARCH_PROFILE=smoke AUTORESEARCH_BACKEND=cpu python prepare.py --smoke --synthetic
```

## Run Commands

CPU smoke test:

```bash
./scripts/run_cpu_smoke.sh
```

TT environment check:

```bash
TT_VISIBLE_DEVICES=0 ./scripts/check_tt_env.sh
```

The TT shell wrappers retry once after a recoverable TT bring-up failure by default on N300 hosts. If you want an unconditional host-side reset before the run, set `AUTORESEARCH_TT_RESET_BEFORE_RUN=1`. Use `AUTORESEARCH_TT_RESET_WAIT_SECS` to change the post-reset wait and `AUTORESEARCH_TT_SMI_TIMEOUT_SECS` to bound `tt-smi` calls on wedged hosts.

60-second TT smoke run:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=smoke AUTORESEARCH_TIME_BUDGET=60 ./scripts/run_tt_smoke.sh
```

300-second TT baseline run:

```bash
AUTORESEARCH_BACKEND=tt AUTORESEARCH_PROFILE=tt_singlechip ./scripts/run_tt_baseline.sh
```

## Profiles

`profile=upstreamish`
- Preserves the original 2048-token context, 5-minute budget, large evaluation budget, and upstream-scale batch geometry as closely as practical.
- This is mainly for semantic comparison, not the default one-device TT path.

`profile=tt_singlechip`
- Default.
- Uses a smaller model and shorter sequence length so a single TT device can complete an honest baseline run.
- Keeps the 300-second training budget, the exact `val_bpb` definition, and the same `results.tsv` workflow.

`profile=smoke`
- Synthetic-data-friendly, short-context, small-model profile for CI and smoke tests.

## Environment Overrides

Required env overrides supported by [`configs.py`](/workdir/autoresearch-tenstorrent/configs.py):

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

Useful extra override:

- `AUTORESEARCH_CACHE_DIR`

## Training-Time Accounting

`training_seconds` is the budgeted time used for experiment comparison. This repo excludes the first few warmup steps from `training_seconds` so compilation and graph warmup do not consume the 60-second or 300-second research budget. `total_seconds` includes everything.

## Output Contract

The final summary prints these keys exactly:

- `backend:`
- `tt_device:`
- `init_val_bpb:`
- `val_bpb:`
- `training_seconds:`
- `total_seconds:`
- `peak_vram_mb:`
- `mfu_percent:`
- `total_tokens_M:`
- `num_steps:`
- `num_params_M:`
- `depth:`
- `tokens_per_sec_avg:`

The repo prints `-1.0` for TT metrics that are not measured honestly.

## Research Loop

- `train.py` is the intended mutation target during autonomous experiments.
- `prepare.py`, `configs.py`, `tt_runtime.py`, scripts, and tests stay frozen during the experiment loop.
- `results.tsv` is tab-separated and initialized with:

```text
commit	val_bpb	memory_gb	status	description
```

See [`program.md`](/workdir/autoresearch-tenstorrent/program.md) for the Tenstorrent-specific protocol.

## Known Limitations

See [`docs/KNOWN_LIMITATIONS.md`](/workdir/autoresearch-tenstorrent/docs/KNOWN_LIMITATIONS.md).

Current highlights:

- Sliding-window attention is implemented but default-off.
- Whole-model `torch.compile(backend="tt")` is experimental and default-off.
- Peak VRAM and MFU are placeholders on TT unless a future change measures them honestly.

## Troubleshooting

If a TT run fails with unsupported ops or lazy graph issues:

- Re-run the smallest repro in [`debug/repro_attention.py`](/workdir/autoresearch-tenstorrent/debug/repro_attention.py).
- Force CPU with `AUTORESEARCH_BACKEND=cpu` to separate correctness bugs from lowering bugs.
- Keep `AUTORESEARCH_ENABLE_SLIDING_WINDOW=0`.
- Keep `AUTORESEARCH_ENABLE_TT_COMPILE=0`.

If a TT run fails before model code with messages like `Read unexpected run_mailbox value from core` or `Timeout waiting for Ethernet core service remote IO request flush`:

- Use [`scripts/check_tt_env.sh`](/workdir/autoresearch-tenstorrent/scripts/check_tt_env.sh), [`scripts/run_tt_smoke.sh`](/workdir/autoresearch-tenstorrent/scripts/run_tt_smoke.sh), or [`scripts/run_tt_baseline.sh`](/workdir/autoresearch-tenstorrent/scripts/run_tt_baseline.sh). They retry once after `tt-smi --reset 0`.
- Set `AUTORESEARCH_TT_INIT_RETRIES`, `AUTORESEARCH_TT_RESET_WAIT_SECS`, or `AUTORESEARCH_TT_SMI_TIMEOUT_SECS` if you need a different recovery policy.
- Set `TT_VISIBLE_DEVICES` before importing `jax` or `torch_xla`.
- Use the TT-XLA eager debugging path from [`tt_runtime.py`](/workdir/autoresearch-tenstorrent/tt_runtime.py) only for diagnosis, not as the training baseline.

If TT-XLA initialization fails:

- Re-run `./scripts/check_tt_env.sh`.
- Confirm `/dev/tenstorrent` exists and `tt-smi -ls` lists hardware.
- On N300, set `TT_VISIBLE_DEVICES=0` before importing `jax` or `torch_xla`. Prefer `TT_VISIBLE_DEVICES` over the older `TT_METAL_VISIBLE_DEVICES`.
- The repo launch scripts retry after a recoverable TT startup failure by default. For direct `python train.py` runs, use `AUTORESEARCH_TT_RESET_BEFORE_INIT=1` if the previous TT process died during startup.
- Confirm the TT-XLA runtime version matches the installed firmware/device stack.
- If the runtime fails during startup with hugepage pinning or NOC-address warnings, fix the host/container hugepage setup before debugging model code. Those failures happen before this repo's training code is involved.
- If startup logs include `Read unexpected run_mailbox value` or `Fabric Router Sync: Timeout`, reset the board, wait for the links to retrain, and retry:

```bash
tt-smi --reset 0
sleep 30
TT_VISIBLE_DEVICES=0 ./scripts/check_tt_env.sh
```

## License

MIT. Upstream-derived logic from `karpathy/autoresearch` is retained under the same license terms.
