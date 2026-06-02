# Week 02 — Tokenization + Embeddings

## What was covered

- Tiny tokenizer built from scratch
- Converting token ids into learned embedding vectors
- Adding sinusoidal positional encodings

## Key insights

- Embedding lookup is just indexing a learned table
- Positional encodings inject order information into otherwise order-agnostic token vectors
- Even simple whitespace tokenization is enough to understand the mechanics

## What was confusing

- Why sinusoidal encodings can extrapolate to longer lengths
- How positional encoding interacts with learned embeddings during training
