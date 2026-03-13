from __future__ import annotations

import dataclasses
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict


def _cache_root() -> Path:
    return Path(os.environ.get("AUTORESEARCH_CACHE_DIR", Path.home() / ".cache" / "autoresearch"))


@dataclass(frozen=True)
class TrainConfig:
    profile: str
    backend: str
    cache_dir: Path
    time_budget: int
    max_seq_len: int
    eval_tokens: int
    depth: int
    aspect_ratio: int
    head_dim: int
    total_batch_size: int
    device_batch_size: int
    eval_batch_size: int
    window_pattern: str
    enable_sliding_window: bool
    enable_tt_compile: bool
    seed: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    warmup_steps: int
    tokenizer_batch_size: int
    train_num_shards: int
    smoke_data: bool
    synthetic_data: bool
    bf16: bool
    experimental_compile_options: Dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return self.cache_dir / "data"

    @property
    def tokenizer_dir(self) -> Path:
        return self.cache_dir / "tokenizer"

    @property
    def tokens_per_step(self) -> int:
        return self.device_batch_size * self.max_seq_len

    @property
    def grad_accum_steps(self) -> int:
        if self.total_batch_size % self.tokens_per_step != 0:
            raise ValueError(
                f"TOTAL_BATCH_SIZE={self.total_batch_size} must be divisible by "
                f"DEVICE_BATCH_SIZE*MAX_SEQ_LEN={self.tokens_per_step}"
            )
        return self.total_batch_size // self.tokens_per_step


UPSTREAMISH = TrainConfig(
    profile="upstreamish",
    backend="auto",
    cache_dir=_cache_root(),
    time_budget=300,
    max_seq_len=2048,
    eval_tokens=40 * 524_288,
    depth=8,
    aspect_ratio=64,
    head_dim=128,
    total_batch_size=2**19,
    device_batch_size=128,
    eval_batch_size=128,
    window_pattern="SSSL",
    enable_sliding_window=False,
    enable_tt_compile=False,
    seed=42,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=10,
    tokenizer_batch_size=128,
    train_num_shards=10,
    smoke_data=False,
    synthetic_data=False,
    bf16=True,
    experimental_compile_options={"optimization_level": 1},
)

TT_SINGLECHIP = TrainConfig(
    profile="tt_singlechip",
    backend="auto",
    cache_dir=_cache_root(),
    time_budget=300,
    max_seq_len=512,
    eval_tokens=4 * 65_536,
    depth=4,
    aspect_ratio=64,
    head_dim=64,
    total_batch_size=2**15,
    device_batch_size=8,
    eval_batch_size=8,
    window_pattern="L",
    enable_sliding_window=False,
    enable_tt_compile=False,
    seed=42,
    learning_rate=6e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=3,
    tokenizer_batch_size=128,
    train_num_shards=10,
    smoke_data=False,
    synthetic_data=False,
    bf16=True,
    experimental_compile_options={"optimization_level": 1},
)

SMOKE = TrainConfig(
    profile="smoke",
    backend="auto",
    cache_dir=_cache_root(),
    time_budget=60,
    max_seq_len=128,
    eval_tokens=8_192,
    depth=2,
    aspect_ratio=64,
    head_dim=64,
    total_batch_size=1_024,
    device_batch_size=4,
    eval_batch_size=4,
    window_pattern="L",
    enable_sliding_window=False,
    enable_tt_compile=False,
    seed=123,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=2,
    tokenizer_batch_size=32,
    train_num_shards=2,
    smoke_data=True,
    synthetic_data=True,
    bf16=False,
    experimental_compile_options={"optimization_level": 0},
)


PROFILES = {
    "upstreamish": UPSTREAMISH,
    "tt_singlechip": TT_SINGLECHIP,
    "smoke": SMOKE,
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_override(name: str, current: Any) -> Any:
    value = os.environ.get(name)
    if value is None:
        return current
    if isinstance(current, bool):
        return _parse_bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value)
    return value


def load_config() -> TrainConfig:
    profile = os.environ.get("AUTORESEARCH_PROFILE", "tt_singlechip")
    if profile not in PROFILES:
        raise ValueError(f"Unknown AUTORESEARCH_PROFILE={profile!r}")
    cfg = replace(PROFILES[profile], profile=profile, cache_dir=_cache_root())

    overrides = {
        "backend": _env_override("AUTORESEARCH_BACKEND", cfg.backend),
        "profile": profile,
        "time_budget": _env_override("AUTORESEARCH_TIME_BUDGET", cfg.time_budget),
        "max_seq_len": _env_override("AUTORESEARCH_MAX_SEQ_LEN", cfg.max_seq_len),
        "eval_tokens": _env_override("AUTORESEARCH_EVAL_TOKENS", cfg.eval_tokens),
        "depth": _env_override("AUTORESEARCH_DEPTH", cfg.depth),
        "total_batch_size": _env_override("AUTORESEARCH_TOTAL_BATCH_SIZE", cfg.total_batch_size),
        "device_batch_size": _env_override("AUTORESEARCH_DEVICE_BATCH_SIZE", cfg.device_batch_size),
        "eval_batch_size": _env_override("AUTORESEARCH_DEVICE_BATCH_SIZE", cfg.eval_batch_size),
        "window_pattern": _env_override("AUTORESEARCH_WINDOW_PATTERN", cfg.window_pattern),
        "enable_sliding_window": _env_override(
            "AUTORESEARCH_ENABLE_SLIDING_WINDOW", cfg.enable_sliding_window
        ),
        "enable_tt_compile": _env_override(
            "AUTORESEARCH_ENABLE_TT_COMPILE", cfg.enable_tt_compile
        ),
        "seed": _env_override("AUTORESEARCH_SEED", cfg.seed),
    }
    return replace(cfg, **overrides)


def config_dict(cfg: TrainConfig) -> Dict[str, Any]:
    values = asdict(cfg)
    values["cache_dir"] = str(cfg.cache_dir)
    values["data_dir"] = str(cfg.data_dir)
    values["tokenizer_dir"] = str(cfg.tokenizer_dir)
    values["grad_accum_steps"] = cfg.grad_accum_steps
    return values


def format_config(cfg: TrainConfig) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(config_dict(cfg).items()))
