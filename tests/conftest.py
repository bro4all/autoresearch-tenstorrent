from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def reload_repo_modules():
    for name in ("configs", "prepare", "train", "tt_runtime"):
        if name in sys.modules:
            del sys.modules[name]
    modules = {}
    for name in ("configs", "prepare", "train", "tt_runtime"):
        modules[name] = importlib.import_module(name)
    return modules


def configure_test_env(monkeypatch, tmp_path: Path, backend: str = "cpu", profile: str = "smoke") -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("AUTORESEARCH_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("AUTORESEARCH_BACKEND", backend)
    monkeypatch.setenv("AUTORESEARCH_PROFILE", profile)
    monkeypatch.setenv("AUTORESEARCH_TIME_BUDGET", "5")
    if backend == "tt":
        monkeypatch.setenv("TT_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("TT_METAL_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("AUTORESEARCH_TT_PREFLIGHT_RESET_ON_FAIL", "1")
        monkeypatch.setenv("AUTORESEARCH_TT_PREFLIGHT_RETRIES", "1")
        monkeypatch.setenv("AUTORESEARCH_TT_LIST_TIMEOUT_SECS", "10")
        monkeypatch.setenv("AUTORESEARCH_TT_RESET_WAIT_SECS", "60")
    return cache_dir


def parse_summary(stdout: str):
    data = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data
