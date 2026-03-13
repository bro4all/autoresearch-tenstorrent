# Known Limitations

## Fast Paths

- Sliding-window attention is implemented but default-off.
- Whole-model `torch.compile(backend="tt")` is experimental and default-off.
- No CUDA-style fused optimizer path is shipped.

## Metrics

- `peak_vram_mb` prints `-1.0` on TT until an honest measurement path is added.
- `mfu_percent` prints `-1.0` on TT; the upstream H100 constant is intentionally not reused.

## Runtime

- The baseline assumes lazy `torch_xla` execution on TT and syncs only at step boundaries or when values are materialized.
- TT runtime setup depends on a matching combination of firmware, TT-XLA runtime, and host/container environment.
- A misconfigured host/container can fail before model code runs with TT-XLA errors around hugepage pinning or NOC-address mapping. In that case, fix the TT runtime environment first.
- On N300, use `TT_VISIBLE_DEVICES=0` before importing `jax` or `torch_xla`. The older `TT_METAL_VISIBLE_DEVICES` selector is not the preferred path.
- After a failed bring-up, the board can need `tt-smi --reset 0` plus a short wait before the remote chip retrains and fabric initialization succeeds again.

## Profiles

- `upstreamish` is for semantic comparison and is not the default one-device TT profile.
- `tt_singlechip` is the default TT baseline.
- `smoke` uses small synthetic/offline-friendly inputs for tests.

## Known Working Intent

- CPU smoke path: supported by the test suite in this repo.
- TT single-device baseline intent: one TT device with TT-XLA runtime installed and `AUTORESEARCH_PROFILE=tt_singlechip`.
- Recommended device family for first validation: Wormhole-class single-device targeting.

## Debugging Guidance

- If TT lowering fails, prefer simplifying the offending PyTorch graph over changing frameworks.
- Use the small repro under [`debug/repro_attention.py`](/workdir/autoresearch-tenstorrent/debug/repro_attention.py) to isolate attention-path issues.
- Use TT-XLA eager mode only for diagnosis.
