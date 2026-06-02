"""Week 01: Dot products, softmax, and toy attention in plain PyTorch."""

from __future__ import annotations

import math
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax implemented from first principles."""
    # Subtract the max before exponentiation to avoid very large exponent values.
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head scaled dot-product attention.

    Shapes:
        query: (..., seq_q, d_k)
        key:   (..., seq_k, d_k)
        value: (..., seq_k, d_v)
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    weights = softmax(scores, dim=-1)
    context = weights @ value
    return context, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 2)

    ctx, attn = scaled_dot_product_attention(q, k, v)
    print("attention weights shape:", attn.shape)
    print("context shape:", ctx.shape)
