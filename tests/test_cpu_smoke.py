from __future__ import annotations

import math

import torch

from tests.conftest import configure_test_env, reload_repo_modules


def test_cpu_smoke(monkeypatch, tmp_path):
    configure_test_env(monkeypatch, tmp_path, backend="cpu", profile="smoke")
    monkeypatch.setenv("AUTORESEARCH_MAX_SEQ_LEN", "96")
    monkeypatch.setenv("AUTORESEARCH_DEPTH", "2")
    monkeypatch.setenv("AUTORESEARCH_TOTAL_BATCH_SIZE", "3072")
    monkeypatch.setenv("AUTORESEARCH_DEVICE_BATCH_SIZE", "4")
    modules = reload_repo_modules()
    cfg = modules["configs"].load_config()
    assert cfg.max_seq_len == 96
    assert cfg.depth == 2
    assert cfg.total_batch_size == 3072
    assert cfg.device_batch_size == 4

    prepare = modules["prepare"]
    prepare.prepare_synthetic_cache()
    tokenizer = prepare.Tokenizer.from_directory()
    encoded = tokenizer.encode("tenstorrent autoresearch", prepend=tokenizer.get_bos_token_id())
    decoded = tokenizer.decode(encoded[1:])
    assert "tenstorrent" in decoded

    train = modules["train"]
    model_cfg = train.build_model_config(cfg, tokenizer.get_vocab_size())
    model = train.GPT(model_cfg)
    model.init_weights()
    idx = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    targets = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    logits = model(idx)
    loss = model(idx, targets)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    val_bpb = prepare.evaluate_bpb(
        model,
        tokenizer,
        batch_size=2,
        device=torch.device("cpu"),
        max_seq_len=cfg.max_seq_len,
        eval_tokens=cfg.eval_tokens,
    )
    assert math.isfinite(val_bpb)
