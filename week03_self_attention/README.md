# Week 03 — Self-Attention in Pure PyTorch

## What was covered

- Building scaled dot-product self-attention without `nn.MultiheadAttention`
- Creating `Q`, `K`, and `V` projections manually
- Applying causal masks for autoregressive decoding intuition

## Key insights

- Attention can be implemented with matrix multiplies and softmax only
- Causal masking prevents tokens from reading future tokens
- Query-key similarity determines where information flows

## What was confusing

- Why transposing key is necessary for score computation
- How masking values (large negative vs `-inf`) affect numerical behavior
