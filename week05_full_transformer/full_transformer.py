"""Week 05: Tiny full transformer pieces + RoPE and ALiBi demonstrations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply a simple Rotary Position Embedding (RoPE) to last dimension pairs.

    Expected shape: (batch, heads, seq, head_dim), with even head_dim.
    """
    b, h, s, d = x.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    positions = torch.arange(s, device=x.device, dtype=x.dtype).unsqueeze(-1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angles = positions * freqs

    cos = torch.cos(angles).view(1, 1, s, d // 2)
    sin = torch.sin(angles).view(1, 1, s, d // 2)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    out = torch.zeros_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


def alibi_bias(seq_q: int, seq_k: int, device: torch.device) -> torch.Tensor:
    """Create a basic ALiBi-style distance bias for a single head."""
    q = torch.arange(seq_q, device=device).unsqueeze(1)
    k = torch.arange(seq_k, device=device).unsqueeze(0)
    distance = (k - q).clamp(min=0)
    slope = -0.25
    return slope * distance.float()


class TinyAttention(nn.Module):
    """Attention module used for both self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        causal: bool = False,
        use_rope: bool = False,
        use_alibi: bool = False,
    ) -> torch.Tensor:
        q = self._split(self.to_q(query_input))
        k = self._split(self.to_k(key_value_input))
        v = self._split(self.to_v(key_value_input))

        if use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        if use_alibi:
            scores = scores + alibi_bias(scores.size(-2), scores.size(-1), scores.device)

        if causal:
            mask = torch.triu(
                torch.ones(scores.size(-2), scores.size(-1), device=scores.device),
                diagonal=1,
            ).bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = scores.softmax(dim=-1)
        return self.out(self._merge(weights @ v))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256) -> None:
        super().__init__()
        self.attn = TinyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, causal=False, use_rope=True))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256) -> None:
        super().__init__()
        self.self_attn = TinyAttention(d_model, n_heads)
        self.cross_attn = TinyAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        y = self.norm1(y + self.self_attn(y, y, causal=True, use_alibi=True))
        y = self.norm2(y + self.cross_attn(y, enc_out, causal=False))
        y = self.norm3(y + self.ff(y))
        return y


class TinySeq2SeqTransformer(nn.Module):
    """A tiny encoder-decoder transformer for educational seq2seq experiments."""

    def __init__(self, vocab_size: int, d_model: int = 64) -> None:
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderLayer(d_model=d_model)
        self.decoder = DecoderLayer(d_model=d_model)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(self.src_embed(src_tokens))
        dec = self.decoder(self.tgt_embed(tgt_tokens), enc)
        return self.to_logits(dec)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinySeq2SeqTransformer(vocab_size=50, d_model=32)

    src = torch.randint(0, 50, (2, 6))
    tgt = torch.randint(0, 50, (2, 5))
    logits = model(src, tgt)
    print("seq2seq logits shape:", logits.shape)
