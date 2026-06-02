"""Week 07: Tiny LoRA example for parameter-efficient fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and trainable low-rank update."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_update = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scaling * lora_update


class TinyTextClassifier(nn.Module):
    """A tiny BERT-style toy classifier using mean pooled token embeddings."""

    def __init__(self, vocab_size: int = 100, d_model: int = 32, n_classes: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = LoRALinear(d_model, d_model, rank=4, alpha=4.0)
        self.act = nn.GELU()
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        x = x.mean(dim=1)  # Mean pool for a simple sentence-level representation.
        x = self.act(self.proj(x))
        return self.head(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyTextClassifier(vocab_size=200, d_model=32, n_classes=3)
    batch_tokens = torch.randint(0, 200, (4, 12))
    logits = model(batch_tokens)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("logits shape:", logits.shape)
    print(f"trainable params: {trainable}/{total}")
