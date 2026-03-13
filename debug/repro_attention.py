from __future__ import annotations

import torch

from train import causal_attention


def main():
    torch.manual_seed(0)
    q = torch.randn(1, 16, 4, 32)
    k = torch.randn(1, 16, 4, 32)
    v = torch.randn(1, 16, 4, 32)
    out = causal_attention(q, k, v, window_size=None)
    print(out.shape, out.abs().mean().item())


if __name__ == "__main__":
    main()
