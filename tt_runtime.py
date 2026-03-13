from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from functools import lru_cache
from typing import Dict, Optional


def _tt_smi_timeout_seconds() -> int:
    return int(os.environ.get("AUTORESEARCH_TT_SMI_TIMEOUT_SECS", "30"))


def _prime_tt_environment() -> None:
    if "TT_VISIBLE_DEVICES" not in os.environ and "TT_METAL_VISIBLE_DEVICES" not in os.environ:
        # On N300, exposing PCIe device 0 also exposes the Ethernet-connected peer.
        os.environ["TT_VISIBLE_DEVICES"] = os.environ.get("AUTORESEARCH_TT_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")


def _import_torch_xla():
    import torch_xla  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.runtime as xr  # type: ignore

    return torch_xla, xm, xr


def _first_visible_device() -> str:
    visible = os.environ.get("TT_VISIBLE_DEVICES") or os.environ.get("TT_METAL_VISIBLE_DEVICES") or "0"
    return visible.split(",")[0].strip() or "0"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _maybe_reset_before_init() -> None:
    if not _env_flag("AUTORESEARCH_TT_RESET_BEFORE_INIT"):
        return
    if not shutil.which("tt-smi"):
        return
    device = os.environ.get("AUTORESEARCH_TT_RESET_DEVICE", _first_visible_device())
    wait_seconds = int(os.environ.get("AUTORESEARCH_TT_RESET_WAIT_SECS", "30"))
    print(f"Resetting Tenstorrent device {device} before TT-XLA init")
    try:
        proc = subprocess.run(
            ["tt-smi", "--reset", device],
            check=False,
            capture_output=True,
            text=True,
            timeout=_tt_smi_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"tt-smi --reset {device} timed out after {exc.timeout}s") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        raise RuntimeError(f"tt-smi --reset {device} failed: {detail}")
    time.sleep(max(wait_seconds, 0))


@lru_cache(maxsize=1)
def tt_runtime_importable() -> bool:
    _prime_tt_environment()
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
            timeout=_tt_smi_timeout_seconds(),
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
    _prime_tt_environment()
    _maybe_reset_before_init()
    if not tt_runtime_importable():
        raise RuntimeError("torch_xla is not importable; install the TT-XLA runtime first")
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
            timeout=_tt_smi_timeout_seconds(),
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
