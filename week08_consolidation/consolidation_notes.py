"""Week 08: Programmatic summary of major transformer lessons."""

from __future__ import annotations


def key_insights() -> list[str]:
    """Return concise insights gathered across the 8-week curriculum."""
    return [
        "Attention = similarity scores + softmax + weighted sum.",
        "Residual connections and layer norm stabilize deep transformer blocks.",
        "Positional information can be injected with sinusoidal, RoPE, or ALiBi schemes.",
        "ViT uses the same transformer logic after converting image patches into tokens.",
        "LoRA enables parameter-efficient fine-tuning by learning low-rank updates.",
    ]


if __name__ == "__main__":
    for idx, insight in enumerate(key_insights(), start=1):
        print(f"{idx}. {insight}")
