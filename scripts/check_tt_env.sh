#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/tt_common.sh"

tt_set_visible_devices
tt_maybe_host_reset

if [[ "${AUTORESEARCH_TT_USE_DOCKER:-0}" == "1" ]]; then
  IMAGE="${AUTORESEARCH_TT_IMAGE:-ghcr.io/tenstorrent/tt-xla-slim:latest}"
  tt_run_with_recovery docker run --rm \
    -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES}" \
    -e PJRT_DEVICE="${PJRT_DEVICE:-TT}" \
    -e XLA_STABLEHLO_COMPILE="${XLA_STABLEHLO_COMPILE:-1}" \
    --device /dev/tenstorrent \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    "${IMAGE}" \
    bash -lc 'python -c "import jax; print(jax.devices(\"tt\"))" && python -c '\''import torch_xla.runtime as xr; import torch_xla.core.xla_model as xm; xr.set_device_type("TT"); print(xm.xla_device())'\'''
  exit 0
fi

tt_run_with_recovery python -c "import jax; print(jax.devices('tt'))"
tt_run_with_recovery python - <<'PY'
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
xr.set_device_type("TT")
print(xm.xla_device())
PY
