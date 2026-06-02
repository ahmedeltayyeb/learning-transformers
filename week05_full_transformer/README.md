# Week 05 — Full Transformer + Positional Encoding Variants

## What was covered

- Building encoder and decoder stacks for a tiny seq2seq model
- Applying causal masking in decoder self-attention
- Exploring both RoPE and ALiBi positional bias ideas

## Key insights

- The full transformer is mostly repeated attention and feedforward blocks
- Decoder uses two attention steps: self-attention then cross-attention
- RoPE rotates features by position, while ALiBi adds distance-aware bias to scores

## What was confusing

- Keeping tensor shapes aligned across encoder-decoder attention
- Choosing where and how to apply RoPE/ALiBi in a clean implementation
