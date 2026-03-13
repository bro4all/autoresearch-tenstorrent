#!/usr/bin/env bash
set -euo pipefail

export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-${AUTORESEARCH_TT_VISIBLE_DEVICES:-0}}"

if [[ "${AUTORESEARCH_TT_USE_DOCKER:-0}" == "1" ]]; then
  IMAGE="${AUTORESEARCH_TT_IMAGE:-ghcr.io/tenstorrent/tt-xla-slim:latest}"
  docker run --rm \
    -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES}" \
    --device /dev/tenstorrent \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    "${IMAGE}" \
    bash -lc 'python -c "import jax; print(jax.devices(\"tt\"))"'
  docker run --rm \
    -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES}" \
    --device /dev/tenstorrent \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    "${IMAGE}" \
    python -c 'import torch_xla.runtime as xr; import torch_xla.core.xla_model as xm; xr.set_device_type("TT"); print(xm.xla_device())'
  exit 0
fi

python -c "import jax; print(jax.devices('tt'))"
python - <<'PY'
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
xr.set_device_type("TT")
print(xm.xla_device())
PY
