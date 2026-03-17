# Porting Notes

This repo preserves the upstream `prepare.py` semantic contract while intentionally changing the training/runtime stack for Tenstorrent.

## Preserved From Upstream

- Cache layout under `~/.cache/autoresearch` by default
- Pinned validation shard `shard_06542.parquet`
- `rustbpe` tokenizer training flow
- `token_bytes.pt` construction
- BOS-packed best-fit dataloader
- Exact `val_bpb` definition: per-token cross-entropy in nats divided by `log(2) * total_bytes`
- Fixed wall-clock training budget workflow
- `results.tsv` experiment logging model

## Intentional Deviations

1. CUDA backend replaced with TT-XLA.
   - Upstream assumes one NVIDIA GPU and uses `torch.cuda.*`.
   - This port uses `torch_xla` lazy execution on TT through [`tt_runtime.py`](../tt_runtime.py).

2. FlashAttention dependency removed.
   - Upstream uses FA3/Hopper-oriented fast paths.
   - This port uses pure PyTorch causal attention so CPU and TT share the same baseline implementation.

3. CUDA fused optimizer path removed.
   - Upstream mixes Muon and fused AdamW.
   - This port uses standard `torch.optim.AdamW` as the baseline.

4. Configs moved out of `train.py`.
   - Upstream edits constants directly in `train.py`.
   - This repo keeps runtime-tunable values in [`configs.py`](../configs.py) so profiles and env overrides are explicit.

5. TT-friendly default profile added.
   - `profile=tt_singlechip` is smaller than the H100-oriented upstream baseline.
   - The goal is correctness and completion on one TT device, not parity with H100 throughput.
   - The currently verified N300-safe default is `max_seq_len=256`, `depth=2`, `total_batch_size=16384`, `device_batch_size=64`, `eval_tokens=262144`, `bf16=0`.

6. Experimental features are opt-in.
   - Sliding-window attention exists but defaults off.
   - Whole-model `torch.compile(backend="tt")` exists but defaults off.

7. TT-only unavailable metrics are placeholders.
   - `peak_vram_mb` and `mfu_percent` print `-1.0` unless measured honestly.

8. Synthetic smoke mode added.
   - `prepare.py --smoke --synthetic` creates tiny local parquet shards for offline tests.
   - This is test scaffolding only; the real baseline path still uses the upstream shard/tokenizer layout.

9. TT-friendly profiles freeze token/value embeddings by default.
   - On the tested N300 + TT-XLA stack, training `transformer.wte` and per-layer `value_embeds` with plain AdamW caused non-finite TT weights within a few steps even though CPU-vs-TT forward correctness passed.
   - `smoke` and `tt_singlechip` therefore freeze those embeddings by default on TT and optimize the rest of the model.
   - `upstreamish` keeps embeddings trainable by default.

10. Host-side TT preflight added for N300 stability.
   - The documented TT entrypoints now require `tt-smi -ls` to succeed on the host before they launch JAX or `torch_xla`.
   - If board management is unhealthy, the wrappers perform a bounded host reset and re-check the board before the Python process starts.
   - Reset recovery now polls for board-management health instead of sleeping a fixed interval, because this N300 host often retrains more slowly after TT runtime/compiler aborts.
   - Reset ownership is intentionally kept in shell wrappers so direct `python train.py` remains simple and container probes stay probe-only.

## Why TT-XLA

- TT-XLA is the documented frontend Tenstorrent recommends for PyTorch.
- TT-Forge-ONNX can ingest PyTorch, but Tenstorrent’s own docs recommend TT-XLA for PyTorch workloads.
- `tt-torch` is deprecated and is intentionally not used here.

## Training Budget Accounting

Upstream excludes startup and compile warmup from the 5-minute comparison window by ignoring early steps. This port keeps that idea and records `training_seconds` only after configurable warmup steps. `total_seconds` still includes the whole run.
