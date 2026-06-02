"""Week 04: Transformer encoder block built from small transparent pieces."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head attention implementation (educational version)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1)
        attended = weights @ v
        return self.out(self._merge_heads(attended))


class EncoderBlock(nn.Module):
    """One encoder block: MHA -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    block = EncoderBlock(d_model=32, n_heads=4, d_ff=64)
    sample = torch.randn(2, 10, 32)
    output = block(sample)
    print("encoder output shape:", output.shape)
