# Lessons

## 2026-03-13

- When `tt-smi -ls` shows Tenstorrent hardware, do not treat TT as unavailable just because `torch_xla` or `jax` are missing or the first container probe fails.
- Separate hardware visibility from runtime bring-up. Diagnose hugepages, container flags, runtime versions, and single-device selection before concluding TT validation is blocked.
- Before claiming a user-configured tool like `gh` is unavailable, search persisted volume paths such as `/workdir`, `/workdir/home`, and common credential helpers instead of checking only the current `PATH`.
