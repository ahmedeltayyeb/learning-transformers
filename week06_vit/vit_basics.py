"""Week 06: Minimal Vision Transformer implementation with patch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(self, image_size: int = 32, patch_size: int = 8, in_chans: int = 3, d_model: int = 64) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution creates one embedding vector per patch location.
        x = self.proj(x)  # (batch, d_model, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        return x


class TinyViT(nn.Module):
    """Educational ViT with CLS token + encoder stack."""

    def __init__(self, num_classes: int = 10, d_model: int = 64, depth: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        patch_tokens = self.patch_embed(x)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        return self.head(cls_out)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyViT(num_classes=5, d_model=32, depth=1, n_heads=4)
    image_batch = torch.randn(2, 3, 32, 32)
    logits = model(image_batch)
    print("vit logits shape:", logits.shape)
