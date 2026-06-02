# Week 01 — Math + Attention Intuition

## What was covered

- Dot products as similarity scores
- Softmax as a way to convert scores into probabilities
- Weighted sum as the core attention operation

## Key insights

- Dot products become larger when vectors point in similar directions
- Softmax amplifies relative differences while keeping values normalized
- Attention is just "score -> normalize -> weighted sum"

## What was confusing

- Why scaling by `sqrt(d_k)` matters for stable gradients
- How score magnitude changes as dimension grows
