# Known Limitations

## Fast Paths

- Sliding-window attention is implemented but default-off.
- Whole-model `torch.compile(backend="tt")` is experimental and default-off.
- BF16 remains default-off on the current N300 stack.
- No CUDA-style fused optimizer path is shipped.

## Metrics

- `peak_vram_mb` prints `-1.0` on TT until an honest measurement path is added.
- `mfu_percent` prints `-1.0` on TT; the upstream H100 constant is intentionally not reused.

## Runtime

- The baseline assumes lazy `torch_xla` execution on TT and syncs only at step boundaries or when values are materialized.
- TT runtime setup depends on a matching combination of firmware, TT-XLA runtime, and host/container environment.
- A misconfigured host/container can fail before model code runs with TT-XLA errors around hugepage pinning or NOC-address mapping. In that case, fix the TT runtime environment first.
- On N300, use `TT_VISIBLE_DEVICES=0` before importing `jax` or `torch_xla`. The older `TT_METAL_VISIBLE_DEVICES` selector is not the preferred path.
- After a failed bring-up, the board can need `tt-smi --reset 0` plus a 30-60 second wait before the remote chip retrains and fabric initialization succeeds again.
- The repo TT shell wrappers now do a host-side `tt-smi -ls` preflight before entering JAX or `torch_xla`. If board management is unhealthy, they retry once after a bounded host reset by default. Tune that behavior with `AUTORESEARCH_TT_PREFLIGHT_RETRIES`, `AUTORESEARCH_TT_LIST_TIMEOUT_SECS`, `AUTORESEARCH_TT_RESET_WAIT_SECS`, and `AUTORESEARCH_TT_SMI_TIMEOUT_SECS`.
- Direct `python train.py` remains supported for clean environments, but it is not the preferred recovery path on a flaky N300 host because it bypasses the shell preflight.
- On the tested N300 stack, TT-XLA can emit `Logical eth core ... is not an active eth core` during init and still recover to a usable `jax.devices('tt')` / `xm.xla_device()` result. Treat the final device probe result as the ground truth.

## Profiles

- `upstreamish` is for semantic comparison and is not the default one-device TT profile.
- `tt_singlechip` is the default TT baseline and freezes token/value embeddings by default.
- The currently verified N300-safe `tt_singlechip` shape is `max_seq_len=256`, `depth=2`, `total_batch_size=16384`, `device_batch_size=64`, `eval_tokens=262144`, `bf16=0`.
- `smoke` uses small synthetic/offline-friendly inputs for tests and also freezes token/value embeddings by default.
- Set `AUTORESEARCH_FREEZE_EMBEDDINGS=0` only if you are intentionally debugging embedding training on your TT-XLA stack.
- On the tested N300 stack, BF16 and whole-model compile were screened on the winning `tt_singlechip` geometry and were not promoted: BF16 hit a TT MLIR compiler abort, and compile mode remained unstable behind board/runtime bring-up failures.

## Known Working Intent

- CPU smoke path: supported by the test suite in this repo.
- TT single-device baseline intent: one TT device with TT-XLA runtime installed and `AUTORESEARCH_PROFILE=tt_singlechip`.
- Measured on the connected N300, the verified 300-second `tt_singlechip` baseline reached `val_bpb` `3.274981 -> 2.421392` in `300.986748` training seconds with `175` steps and `9526.000778` average tokens/sec.
- Recommended device family for first validation: Wormhole-class single-device targeting.

## Debugging Guidance

- If TT lowering fails, prefer simplifying the offending PyTorch graph over changing frameworks.
- Use the small repro under [`debug/repro_attention.py`](../debug/repro_attention.py) to isolate attention-path issues.
- Use TT-XLA eager mode only for diagnosis.
