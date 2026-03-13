from __future__ import annotations

import math

import pytest
import torch

from tests.conftest import configure_test_env, reload_repo_modules


@pytest.mark.tt_hw
def test_tiny_cpu_vs_tt(monkeypatch, tmp_path):
    configure_test_env(monkeypatch, tmp_path, backend="tt", profile="smoke")
    monkeypatch.setenv("AUTORESEARCH_MAX_SEQ_LEN", "128")
    monkeypatch.setenv("AUTORESEARCH_DEPTH", "2")
    modules = reload_repo_modules()
    if not modules["tt_runtime"].tt_runtime_importable():
        pytest.skip("torch_xla is not installed")
    if not modules["tt_runtime"].tt_hardware_available():
        pytest.skip("No Tenstorrent device detected")

    prepare = modules["prepare"]
    train = modules["train"]
    prepare.prepare_synthetic_cache()
    tokenizer = prepare.Tokenizer.from_directory()
    cfg = modules["configs"].load_config()
    model_cfg = train.build_model_config(cfg, tokenizer.get_vocab_size())

    cpu_model = train.GPT(model_cfg)
    cpu_model.init_weights()
    tt_model = train.GPT(model_cfg)
    tt_model.load_state_dict(cpu_model.state_dict())
    device = modules["tt_runtime"].init_tt_device()
    tt_model = tt_model.to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(cfg.seed)
    x_cpu = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    y_cpu = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    logits_cpu = cpu_model(x_cpu)
    loss_cpu = cpu_model(x_cpu, y_cpu)

    x_tt = x_cpu.to(device)
    y_tt = y_cpu.to(device)
    logits_tt = tt_model(x_tt).to("cpu").float()
    loss_tt = tt_model(x_tt, y_tt).to("cpu").float()

    cpu_flat = logits_cpu.flatten().float()
    tt_flat = logits_tt.flatten().float()
    pcc = torch.corrcoef(torch.stack([cpu_flat, tt_flat]))[0, 1].item()
    max_abs_diff = (logits_cpu - logits_tt).abs().max().item()
    mean_abs_diff = (logits_cpu - logits_tt).abs().mean().item()
    loss_diff = abs(loss_cpu.item() - loss_tt.item())

    assert torch.isfinite(logits_tt).all()
    assert math.isfinite(loss_tt.item())
    assert pcc >= 0.98
    assert max_abs_diff <= 0.15
    assert mean_abs_diff <= 0.03
    assert loss_diff <= 0.05
