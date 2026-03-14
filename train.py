"""
Autoresearch pretraining baseline for CPU and Tenstorrent TT-XLA.

The correctness-first path uses pure PyTorch attention and a standard AdamW
optimizer. TT-specific optimizations stay behind flags.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import TrainConfig, config_dict, load_config
from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    EVAL_TOKENS,
    Tokenizer,
    evaluate_bpb,
    make_dataloader,
    prepare_synthetic_cache,
)
from tt_runtime import (
    get_backend,
    get_device_string,
    init_tt_device,
    maybe_set_tt_compile_options,
    optimizer_step,
    sync,
)


DEFAULT_MLP_EXPANSION = 4
LOGIT_SOFTCAP = 15.0
_ATTENTION_MASK_CACHE: Dict[Tuple[str, int, int], torch.Tensor] = {}


@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str
    enable_sliding_window: bool
    mlp_expansion: int = DEFAULT_MLP_EXPANSION


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def get_attention_mask(seq_len: int, window_size: Optional[int], device: torch.device) -> torch.Tensor:
    key = (str(device), seq_len, -1 if window_size is None else int(window_size))
    cached = _ATTENTION_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    if window_size is not None and window_size > 0 and window_size < seq_len:
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        causal_mask = causal_mask | (distance >= window_size)
    _ATTENTION_MASK_CACHE[key] = causal_mask
    return causal_mask


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: Optional[int]) -> torch.Tensor:
    bsz, seq_len, num_heads, head_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
    causal_mask = get_attention_mask(seq_len, window_size, scores.device)
    min_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(causal_mask, min_value)
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(32, self.n_embd)
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos_sin,
        window_size: Optional[int],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = causal_attention(q, k, v, window_size)
        y = y.view(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_expansion * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], cos_sin, window_size: Optional[int]) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        )
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer["wte"].weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        scale = (3.0**0.5) * (self.config.n_embd ** -0.5)
        for block in self.transformer["h"]:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for embedding in self.value_embeds.values():
            torch.nn.init.uniform_(embedding.weight, -scale, scale)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 10_000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config: GPTConfig):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = max(1, long_window // 2)
        windows = []
        for layer_idx in range(config.n_layer):
            if not config.enable_sliding_window:
                windows.append(None)
                continue
            char = pattern[layer_idx % len(pattern)]
            windows.append(long_window if char == "L" else short_window)
        if windows:
            windows[-1] = None
        return windows

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, reduction: str = "mean"):
        batch_size, seq_len = idx.shape
        cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]
        x = self.transformer["wte"](idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
        if targets is None:
            return logits
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view(batch_size, seq_len)
        if reduction == "none":
            return flat_loss
        if reduction == "sum":
            return flat_loss.sum()
        return flat_loss.mean()


def build_model_config(cfg: TrainConfig, vocab_size: int) -> GPTConfig:
    base_dim = cfg.depth * cfg.aspect_ratio
    model_dim = ((base_dim + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
    num_heads = max(1, model_dim // cfg.head_dim)
    return GPTConfig(
        sequence_len=cfg.max_seq_len,
        vocab_size=vocab_size,
        n_layer=cfg.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=cfg.window_pattern,
        enable_sliding_window=cfg.enable_sliding_window,
    )


def maybe_compile_model(model: nn.Module, cfg: TrainConfig, backend: str) -> nn.Module:
    if not cfg.enable_tt_compile or backend != "tt":
        return model
    maybe_set_tt_compile_options(cfg.experimental_compile_options)
    return torch.compile(model, backend="tt")


def maybe_freeze_tt_embeddings(model: GPT, cfg: TrainConfig, backend: str) -> None:
    if backend != "tt" or not cfg.freeze_embeddings:
        return
    model.transformer["wte"].weight.requires_grad_(False)
    for embedding in model.value_embeds.values():
        embedding.weight.requires_grad_(False)


def ensure_data_ready(cfg: TrainConfig) -> None:
    if (cfg.tokenizer_dir / "tokenizer.pkl").exists() and (cfg.data_dir / "shard_06542.parquet").exists():
        return
    if cfg.synthetic_data:
        prepare_synthetic_cache(cfg.cache_dir, seed=cfg.seed)
        return
    raise RuntimeError(
        f"Missing prepared data under {cfg.cache_dir}. Run `python prepare.py` first "
        f"(or `python prepare.py --smoke --synthetic` for smoke tests)."
    )


def _git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "uncommitted"


def append_results_row(summary: Dict[str, float | str], status: str, description: str) -> None:
    path = Path("results.tsv")
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    memory_gb = -1.0 if float(summary["peak_vram_mb"]) < 0 else float(summary["peak_vram_mb"]) / 1024.0
    row = (
        f"{_git_short_hash()}\t{float(summary['val_bpb']):.6f}\t{memory_gb:.1f}\t"
        f"{status}\t{description}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def run_training(cfg: TrainConfig, experiment: bool = False, description: str = "baseline") -> Dict[str, float | str]:
    t_start = time.time()
    backend = get_backend(cfg.backend)
    if cfg.backend == "tt" and backend != "tt":
        raise RuntimeError("AUTORESEARCH_BACKEND=tt was requested but TT runtime/hardware is unavailable")
    device = init_tt_device() if backend == "tt" else torch.device("cpu")
    tt_device = get_device_string(device)

    if cfg.synthetic_data:
        prepare_synthetic_cache(cfg.cache_dir, seed=cfg.seed)
    ensure_data_ready(cfg)

    torch.manual_seed(cfg.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    model_cfg = build_model_config(cfg, vocab_size)
    print(f"Config profile: {cfg.profile}")
    for key, value in sorted(config_dict(cfg).items()):
        print(f"cfg.{key}: {value}")
    print(f"Model config: {asdict(model_cfg)}")

    model = GPT(model_cfg)
    model.init_weights()
    model_dtype = torch.bfloat16 if (backend == "tt" and cfg.bf16) else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    maybe_freeze_tt_embeddings(model, cfg, backend)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    model = maybe_compile_model(model, cfg, backend)

    train_loader = make_dataloader(
        tokenizer,
        cfg.device_batch_size,
        cfg.max_seq_len,
        "train",
        device=device,
        tokenizer_batch_size=cfg.tokenizer_batch_size,
    )
    init_val_bpb = evaluate_bpb(
        model,
        tokenizer,
        cfg.eval_batch_size,
        device=device,
        max_seq_len=cfg.max_seq_len,
        eval_tokens=cfg.eval_tokens,
    )
    print(f"Initial val_bpb: {init_val_bpb:.6f}")

    total_training_time = 0.0
    step = 0
    tokens_processed = 0
    smooth_train_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while True:
        t0 = time.time()
        train_loss = None
        for _ in range(cfg.grad_accum_steps):
            x, y, epoch = next(train_loader)
            loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}: {loss.item()}")
            train_loss = loss.detach()
            (loss / cfg.grad_accum_steps).backward()
        optimizer_step(optimizer, backend)
        optimizer.zero_grad(set_to_none=True)
        sync(backend)
        dt = time.time() - t0
        if step >= cfg.warmup_steps:
            total_training_time += dt
        step += 1
        tokens_processed += cfg.total_batch_size

        train_loss_value = float(train_loss.item()) if train_loss is not None else float("nan")
        if math.isnan(train_loss_value) or train_loss_value > 100.0:
            raise RuntimeError(f"Exploding training loss at step {step}: {train_loss_value}")
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_value
        debiased = smooth_train_loss / (1 - 0.9**step)
        progress = min(total_training_time / max(cfg.time_budget, 1), 1.0)
        tok_per_sec = cfg.total_batch_size / max(dt, 1e-6)
        remaining = max(0.0, cfg.time_budget - total_training_time)
        print(
            f"\rstep {step:05d} ({100*progress:5.1f}%) | loss: {debiased:.6f} | "
            f"dt: {dt*1000:.0f}ms | tok/sec: {int(tok_per_sec):,} | epoch: {epoch} | "
            f"remaining: {remaining:.0f}s",
            end="",
            flush=True,
        )
        if step > cfg.warmup_steps and total_training_time >= cfg.time_budget:
            break
    print()

    val_bpb = evaluate_bpb(
        model,
        tokenizer,
        cfg.eval_batch_size,
        device=device,
        max_seq_len=cfg.max_seq_len,
        eval_tokens=cfg.eval_tokens,
    )
    total_seconds = time.time() - t_start
    summary: Dict[str, float | str] = {
        "backend": backend,
        "tt_device": tt_device,
        "init_val_bpb": float(init_val_bpb),
        "val_bpb": float(val_bpb),
        "training_seconds": float(total_training_time),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": -1.0,
        "mfu_percent": -1.0,
        "total_tokens_M": tokens_processed / 1e6,
        "num_steps": step,
        "num_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "depth": cfg.depth,
        "tokens_per_sec_avg": tokens_processed / max(total_training_time, 1e-6),
    }
    print("---")
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
        value = summary[key]
        if isinstance(value, str):
            print(f"{key}: {value}")
        elif isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")
    if experiment:
        status = "keep" if summary["val_bpb"] <= summary["init_val_bpb"] else "discard"
        append_results_row(summary, status=status, description=description)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoresearch-tenstorrent baseline")
    parser.add_argument("--experiment", action="store_true", help="Append a row to results.tsv")
    parser.add_argument("--description", default="baseline", help="results.tsv description")
    args = parser.parse_args()
    cfg = load_config()
    run_training(cfg, experiment=args.experiment, description=args.description)


if __name__ == "__main__":
    main()
