from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import load_config
from prepare import Tokenizer, prepare_synthetic_cache
from train import GPT, build_model_config
from tt_runtime import init_tt_device, sync


@dataclass
class StepMetrics:
    step: int
    loss_cpu: float
    loss_tt: float
    loss_gap: float
    max_param_diff: float
    mean_param_diff: float
    worst_param: str
    nonfinite_params: list[str]


def _param_diff(cpu_model: GPT, tt_model: GPT) -> tuple[float, float, str, list[str]]:
    max_diff = 0.0
    mean_diff = 0.0
    worst_param = ""
    count = 0
    nonfinite: list[str] = []
    for (name_cpu, param_cpu), (name_tt, param_tt) in zip(cpu_model.named_parameters(), tt_model.named_parameters()):
        assert name_cpu == name_tt
        tt_cpu = param_tt.detach().to("cpu").float()
        cpu_float = param_cpu.detach().float()
        if not torch.isfinite(tt_cpu).all():
            nonfinite.append(name_tt)
        delta = (cpu_float - tt_cpu).abs()
        param_max = float(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).max())
        if param_max >= max_diff:
            max_diff = param_max
            worst_param = name_tt
        finite_mask = torch.isfinite(delta)
        if finite_mask.any():
            mean_diff += float(delta[finite_mask].mean())
            count += 1
    return max_diff, mean_diff / max(count, 1), worst_param, nonfinite


def _freeze_special_params(model: GPT, freeze_scalars: bool, freeze_embeddings: bool) -> None:
    if freeze_scalars:
        model.resid_lambdas.requires_grad_(False)
        model.x0_lambdas.requires_grad_(False)
    if freeze_embeddings:
        model.transformer["wte"].weight.requires_grad_(False)
        for value_embed in model.value_embeds.values():
            value_embed.weight.requires_grad_(False)


def run_compare(
    steps: int,
    learning_rate: float,
    freeze_scalars: bool,
    freeze_embeddings: bool,
) -> list[StepMetrics]:
    cfg = load_config()
    prepare_synthetic_cache(cfg.cache_dir, seed=cfg.seed)
    tokenizer = Tokenizer.from_directory()
    model_cfg = build_model_config(cfg, tokenizer.get_vocab_size())

    torch.manual_seed(cfg.seed)
    cpu_model = GPT(model_cfg)
    cpu_model.init_weights()
    tt_model = GPT(model_cfg)
    tt_model.load_state_dict(cpu_model.state_dict())

    device = init_tt_device()
    tt_model = tt_model.to(device=device, dtype=torch.float32)

    _freeze_special_params(cpu_model, freeze_scalars, freeze_embeddings)
    _freeze_special_params(tt_model, freeze_scalars, freeze_embeddings)

    cpu_params = [p for p in cpu_model.parameters() if p.requires_grad]
    tt_params = [p for p in tt_model.parameters() if p.requires_grad]
    cpu_opt = torch.optim.AdamW(cpu_params, lr=learning_rate)
    tt_opt = torch.optim.AdamW(tt_params, lr=learning_rate)

    x_cpu = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    y_cpu = torch.randint(0, tokenizer.get_vocab_size(), (2, cfg.max_seq_len))
    x_tt = x_cpu.to(device)
    y_tt = y_cpu.to(device)

    metrics: list[StepMetrics] = []
    for step in range(1, steps + 1):
        cpu_loss = cpu_model(x_cpu, y_cpu)
        tt_loss = tt_model(x_tt, y_tt)
        cpu_loss.backward()
        tt_loss.backward()
        cpu_opt.step()
        tt_opt.step()
        cpu_opt.zero_grad(set_to_none=True)
        tt_opt.zero_grad(set_to_none=True)
        sync("tt")
        max_diff, mean_diff, worst_param, nonfinite = _param_diff(cpu_model, tt_model)
        loss_cpu = float(cpu_loss.detach())
        loss_tt = float(tt_loss.detach().to("cpu"))
        metrics.append(
            StepMetrics(
                step=step,
                loss_cpu=loss_cpu,
                loss_tt=loss_tt,
                loss_gap=abs(loss_cpu - loss_tt),
                max_param_diff=max_diff,
                mean_param_diff=mean_diff,
                worst_param=worst_param,
                nonfinite_params=nonfinite,
            )
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare small CPU and TT training trajectories")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-scalars", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    args = parser.parse_args()

    results = run_compare(
        args.steps,
        args.learning_rate,
        args.freeze_scalars,
        args.freeze_embeddings,
    )
    for item in results:
        print(
            f"step={item.step} "
            f"loss_cpu={item.loss_cpu:.6f} "
            f"loss_tt={item.loss_tt:.6f} "
            f"loss_gap={item.loss_gap:.6f} "
            f"max_param_diff={item.max_param_diff:.6f} "
            f"mean_param_diff={item.mean_param_diff:.6f} "
            f"worst_param={item.worst_param or '-'} "
            f"nonfinite={','.join(item.nonfinite_params) if item.nonfinite_params else '-'}"
        )
    final = results[-1]
    if not math.isfinite(final.loss_tt):
        raise SystemExit("TT loss became non-finite")


if __name__ == "__main__":
    main()
