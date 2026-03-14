from __future__ import annotations

import os
import subprocess

import pytest
import torch

from tests.conftest import REPO_ROOT, configure_test_env, parse_summary, reload_repo_modules


def _prepare_cache(monkeypatch, tmp_path, backend: str, profile: str):
    configure_test_env(monkeypatch, tmp_path, backend=backend, profile=profile)
    modules = reload_repo_modules()
    if not modules["tt_runtime"].tt_runtime_importable():
        pytest.skip("torch_xla is not installed")
    if not modules["tt_runtime"].tt_hardware_available():
        pytest.skip("No Tenstorrent device detected")
    modules["prepare"].prepare_synthetic_cache()
    return modules


@pytest.mark.tt_hw
def test_one_step_train(monkeypatch, tmp_path):
    modules = _prepare_cache(monkeypatch, tmp_path, backend="tt", profile="smoke")
    cfg = modules["configs"].load_config()
    tokenizer = modules["prepare"].Tokenizer.from_directory()
    device = modules["tt_runtime"].init_tt_device()
    model_cfg = modules["train"].build_model_config(cfg, tokenizer.get_vocab_size())
    model = modules["train"].GPT(model_cfg)
    model.init_weights()
    model = model.to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loader = modules["prepare"].make_dataloader(
        tokenizer,
        cfg.device_batch_size,
        cfg.max_seq_len,
        "train",
        device=device,
    )
    x, y, _ = next(loader)
    loss_before = model(x, y)
    assert torch.isfinite(loss_before)
    loss_before.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    modules["tt_runtime"].sync("tt")
    loss_after = model(x, y)
    assert torch.isfinite(loss_after)

def _run_script(script_name: str, extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(
        [str(REPO_ROOT / "scripts" / script_name)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.tt_hw
@pytest.mark.long
def test_60s_smoke_run(monkeypatch, tmp_path):
    modules = _prepare_cache(monkeypatch, tmp_path, backend="tt", profile="smoke")
    proc = subprocess.run(
        [str(REPO_ROOT / "scripts" / "run_tt_smoke.sh"), "--experiment", "--description", "smoke-baseline"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "AUTORESEARCH_BACKEND": "tt",
            "AUTORESEARCH_PROFILE": "smoke",
            "AUTORESEARCH_TIME_BUDGET": "60",
            "AUTORESEARCH_CACHE_DIR": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = parse_summary(proc.stdout)
    for key in (
        "backend",
        "tt_device",
        "init_val_bpb",
        "val_bpb",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "mfu_percent",
        "total_tokens_M",
        "num_steps",
        "num_params_M",
        "depth",
        "tokens_per_sec_avg",
    ):
        assert key in summary
    assert 55 <= float(summary["training_seconds"]) <= 75
    assert float(summary["total_seconds"]) <= 180
    assert float(summary["num_steps"]) >= 5
    assert float(summary["val_bpb"]) < float(summary["init_val_bpb"])
    assert (REPO_ROOT / "results.tsv").read_text().count("\n") >= 2


@pytest.mark.tt_hw
@pytest.mark.long
def test_300s_baseline_run(monkeypatch, tmp_path):
    modules = _prepare_cache(monkeypatch, tmp_path, backend="tt", profile="tt_singlechip")
    proc = _run_script(
        "run_tt_baseline.sh",
        {
            "AUTORESEARCH_BACKEND": "tt",
            "AUTORESEARCH_PROFILE": "tt_singlechip",
            "AUTORESEARCH_TIME_BUDGET": "300",
            "AUTORESEARCH_CACHE_DIR": str(tmp_path / "cache"),
        },
    )
    assert proc.returncode == 0, proc.stderr
    summary = parse_summary(proc.stdout)
    assert 295 <= float(summary["training_seconds"]) <= 330
    assert float(summary["total_seconds"]) <= 600
    assert float(summary["num_steps"]) >= 10
    assert float(summary["val_bpb"]) <= float(summary["init_val_bpb"]) - 0.01


@pytest.mark.tt_hw
@pytest.mark.long
def test_reproducibility(monkeypatch, tmp_path):
    modules = _prepare_cache(monkeypatch, tmp_path, backend="tt", profile="smoke")
    common_env = {
        **os.environ,
        "AUTORESEARCH_BACKEND": "tt",
        "AUTORESEARCH_PROFILE": "smoke",
        "AUTORESEARCH_TIME_BUDGET": "60",
        "AUTORESEARCH_SEED": "777",
        "AUTORESEARCH_CACHE_DIR": str(tmp_path / "cache"),
    }
    proc1 = _run_script("run_tt_smoke.sh", common_env)
    proc2 = _run_script("run_tt_smoke.sh", common_env)
    assert proc1.returncode == 0, proc1.stderr
    assert proc2.returncode == 0, proc2.stderr
    s1 = parse_summary(proc1.stdout)
    s2 = parse_summary(proc2.stdout)
    assert abs(float(s1["val_bpb"]) - float(s2["val_bpb"])) <= 0.02
    step_delta = abs(float(s1["num_steps"]) - float(s2["num_steps"])) / max(float(s1["num_steps"]), 1.0)
    assert step_delta <= 0.10
