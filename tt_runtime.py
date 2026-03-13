from __future__ import annotations

import contextlib
import os
import subprocess
from functools import lru_cache
from typing import Dict, Optional


def _import_torch_xla():
    import torch_xla  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.runtime as xr  # type: ignore

    return torch_xla, xm, xr


@lru_cache(maxsize=1)
def tt_runtime_importable() -> bool:
    try:
        _import_torch_xla()
        return True
    except Exception:
        return False


def tt_hardware_available() -> bool:
    if os.path.exists("/dev/tenstorrent/0"):
        return True
    try:
        proc = subprocess.run(
            ["tt-smi", "-ls"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and "Wormhole" in proc.stdout
    except Exception:
        return False


def get_backend(requested: Optional[str] = None) -> str:
    backend = requested or os.environ.get("AUTORESEARCH_BACKEND", "auto")
    backend = backend.lower()
    if backend not in {"auto", "cpu", "tt"}:
        raise ValueError(f"Invalid backend: {backend}")
    if backend == "auto":
        return "tt" if (tt_hardware_available() and tt_runtime_importable()) else "cpu"
    return backend


def init_tt_device():
    if not tt_runtime_importable():
        raise RuntimeError("torch_xla is not importable; install the TT-XLA runtime first")
    if "TT_VISIBLE_DEVICES" not in os.environ and "TT_METAL_VISIBLE_DEVICES" not in os.environ:
        # N300 exposes a PCIe-visible chip plus its Ethernet-connected peer.
        os.environ["TT_VISIBLE_DEVICES"] = os.environ.get("AUTORESEARCH_TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
    _, xm, xr = _import_torch_xla()
    xr.set_device_type("TT")
    device = xm.xla_device()
    print(f"Using TT device: {device}")
    return device


def sync(backend: Optional[str] = None) -> None:
    if get_backend(backend) != "tt":
        return
    torch_xla, _, _ = _import_torch_xla()
    torch_xla.sync(wait=True)


def get_device_string(device=None) -> str:
    backend = get_backend()
    if backend != "tt":
        return "cpu"
    parts = []
    if device is not None:
        parts.append(str(device))
    try:
        proc = subprocess.run(
            ["tt-smi", "-ls"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            summary = " ".join(line.strip() for line in proc.stdout.splitlines() if "Wormhole" in line)
            if summary:
                parts.append(summary)
    except Exception:
        pass
    return " | ".join(dict.fromkeys(parts)) if parts else "tt"


def maybe_set_tt_compile_options(options: Optional[Dict[str, object]] = None) -> None:
    if not options or get_backend() != "tt" or not tt_runtime_importable():
        return
    torch_xla, _, _ = _import_torch_xla()
    torch_xla.set_custom_compile_options(options)


def eager_debug_context(enabled: bool = False):
    if not enabled or get_backend() != "tt" or not tt_runtime_importable():
        return contextlib.nullcontext()
    try:
        from torch_xla.experimental.eager import eager_mode_context  # type: ignore

        return eager_mode_context(True)
    except Exception:
        return contextlib.nullcontext()


def codegen_debug_options(export_path: str) -> Dict[str, object]:
    return {
        "backend": "codegen_py",
        "export_path": export_path,
        "export_tensors": True,
    }
