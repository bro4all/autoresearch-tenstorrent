from __future__ import annotations

import os
import subprocess

import pytest

from tt_runtime import init_tt_device, tt_hardware_available, tt_runtime_importable


@pytest.mark.tt_hw
def test_tt_env():
    if not tt_runtime_importable():
        pytest.skip("torch_xla is not installed in this environment")
    if not tt_hardware_available():
        pytest.skip("No Tenstorrent device detected")
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TT_METAL_VISIBLE_DEVICES", os.environ["TT_VISIBLE_DEVICES"])
    os.environ.setdefault("AUTORESEARCH_TT_RESET_BEFORE_INIT", "0")
    devices_visible = False
    try:
        import jax

        devices = jax.devices("tt")
        devices_visible = len(devices) >= 1
    except Exception:
        proc = subprocess.run(["tt-smi", "-ls"], capture_output=True, text=True, check=False)
        devices_visible = proc.returncode == 0 and "Wormhole" in proc.stdout
    assert devices_visible
    device = init_tt_device()
    assert device is not None
