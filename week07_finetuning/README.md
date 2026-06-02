# Week 07 — Fine-Tuning + LoRA

## What was covered

- Fine-tuning a small transformer-like classifier head
- Freezing most base weights to reduce trainable parameters
- Injecting Low-Rank Adaptation (LoRA) into linear layers

## Key insights

- LoRA keeps base model frozen while learning lightweight low-rank updates
- Parameter-efficient fine-tuning is practical on limited hardware
- Small targeted adapters can recover meaningful task performance

## What was confusing

- Picking LoRA rank and scaling values in a principled way
- Deciding which layers should receive adapters for best trade-offs
