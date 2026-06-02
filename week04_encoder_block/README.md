# Week 04 — Encoder Block

## What was covered

- Multi-head attention built from tensor reshaping and matrix multiplication
- Position-wise feedforward network
- Residual connections + layer normalization

## Key insights

- Multi-head attention is parallel single-head attention over split feature subspaces
- Residual paths help preserve and refine information through depth
- LayerNorm stabilizes activations and training dynamics

## What was confusing

- Correctly reshaping between `(batch, seq, d_model)` and per-head formats
- Why "Add & Norm" appears twice in each encoder block
