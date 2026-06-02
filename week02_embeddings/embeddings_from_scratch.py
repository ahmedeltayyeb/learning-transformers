"""Week 02: Tiny tokenizer, embedding lookup, and sinusoidal positions."""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn as nn


class TinyTokenizer:
    """A minimal word-level tokenizer using whitespace splitting."""

    def __init__(self, texts: list[str]) -> None:
        counts = Counter(token for text in texts for token in text.lower().split())
        vocab = ["<pad>", "<unk>"] + sorted(counts)
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = vocab

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in text.lower().split()]


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Classic sinusoidal encoding from the original Transformer paper."""
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    even_dims = torch.arange(0, d_model, 2, dtype=torch.float32)
    angle_rates = torch.exp(-math.log(10000.0) * even_dims / d_model)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(positions * angle_rates)
    pe[:, 1::2] = torch.cos(positions * angle_rates)
    return pe


if __name__ == "__main__":
    corpus = ["attention is all you need", "transformers are sequence models"]
    tokenizer = TinyTokenizer(corpus)
    token_ids = tokenizer.encode("attention are powerful")

    embedding = nn.Embedding(num_embeddings=len(tokenizer.itos), embedding_dim=8)
    ids = torch.tensor(token_ids)
    token_vectors = embedding(ids)
    pos = sinusoidal_positional_encoding(seq_len=ids.size(0), d_model=8)

    combined = token_vectors + pos
    print("token ids:", token_ids)
    print("combined embedding shape:", combined.shape)
