"""Week 03: Self-attention from first principles (no nn.MultiheadAttention)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Single-head self-attention with explicit Q/K/V linear projections."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if causal:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = scores.softmax(dim=-1)
        return weights @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SelfAttention(d_model=16)
    x = torch.randn(2, 5, 16)
    y = model(x, causal=True)
    print("output shape:", y.shape)
