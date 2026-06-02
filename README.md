# Learning Transformers From First Principles

This repository documents an 8-week learning journey to understand transformer architecture from the ground up.

## Goals

- Build intuition for attention, embeddings, and sequence modeling
- Implement each transformer building block directly in PyTorch
- Connect NLP transformer ideas to Vision Transformers (ViT)
- Practice practical fine-tuning workflows, including LoRA

## How to navigate

Each `weekXX_*` folder contains:

1. `README.md` with what was covered, key insights, and confusing parts
2. One or more Python files with educational, heavily commented implementations

## Weekly roadmap

- `week01_math_attention/` — dot products, softmax, attention intuition, toy PyTorch code
- `week02_embeddings/` — tokenizer from scratch, embedding lookup, positional encoding
- `week03_self_attention/` — scaled dot-product attention in pure PyTorch (no `nn.MultiheadAttention`)
- `week04_encoder_block/` — multi-head attention + feedforward block, layer norm, residual connections
- `week05_full_transformer/` — full encoder/decoder, seq2seq task, RoPE and ALiBi positional encoding
- `week06_vit/` — Vision Transformer (ViT), patch embeddings, CLS token, bridge to CV intuition
- `week07_finetuning/` — fine-tuning a small ViT/BERT-style model and LoRA experimentation
- `week08_consolidation/` — final write-up, polished notes, and key insights

## Prerequisites

- Python 3.10+
- PyTorch

Most files are runnable as small demos with synthetic data.
