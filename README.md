# 🤖 Learning Transformers — From First Principles

A hands-on, from-scratch journey through transformer architecture — built by an ML Engineer with a Computer Vision background who finally decided to stop treating attention as a black box.

> **Goal:** Understand transformers deeply enough to implement, fine-tune, and reason about them in production CV and multimodal systems.

---

## 🗺️ Structure

Each week has its own folder with:
- A `README.md` summarizing what was covered, key insights, and what was still confusing
- One or more heavily commented PyTorch implementations

```
learning-transformers/
├── week01_math_attention/       # Dot products, softmax, attention intuition
├── week02_embeddings/           # Tokenization, embedding lookup, positional encoding
├── week03_self_attention/       # Scaled dot-product attention from scratch
├── week04_encoder_block/        # Multi-head attention + feedforward block
├── week05_full_transformer/     # Full encoder/decoder, seq2seq, RoPE & ALiBi
├── week06_vit/                  # Vision Transformer — where CV meets transformers
├── week07_finetuning/           # Fine-tuning ViT/BERT, LoRA experimentation
└── week08_consolidation/        # Final summary, polished writeup, key insights
```

---

## 📅 Progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Math & Attention Intuition | 🟢 Solid |
| 2 | Embeddings & Tokenization | 🔴 Not started |
| 3 | Self-Attention from Scratch | 🔴 Not started |
| 4 | Multi-Head Attention + Encoder Block | 🔴 Not started |
| 5 | Full Transformer + PE Variants | 🔴 Not started |
| 6 | Vision Transformers (ViT) | 🔴 Not started |
| 7 | Fine-tuning | 🔴 Not started |
| 8 | Consolidation & Publish | 🔴 Not started |

*Status: 🔴 Not started · 🟡 In progress · 🟠 Confused · 🟢 Solid*

---

## 🧠 Philosophy

- **No black boxes.** Every component is implemented from scratch before using library abstractions.
- **Confusion is documented.** Weekly READMEs include what didn't click — not just what did.
- **CV-first perspective.** Theory is always connected back to practical CV/multimodal applications.
- **Messy is fine.** This is a learning repo, not a polished library. Expect experiments, dead ends, and notes-to-self.

---

## 🛠️ Setup

```bash
git clone https://github.com/ahmedeltayyeb/learning-transformers.git
cd learning-transformers
pip install torch torchvision
```

No heavy dependencies beyond PyTorch. Each week may add lightweight extras (e.g. `transformers`, `datasets`) — check the weekly README.

---

## 📚 Core Resources

| Resource | Why |
|----------|-----|
| [3Blue1Brown — Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) | Best visual intuition for attention |
| [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Best hands-on walkthrough |
| [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Best static visual reference |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The original paper |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | Transformers meet CV |

---

## 🔗 Companion Notion Page

Daily logs, concept confidence tracker, and weekly reflections are tracked in Notion alongside this repo.

---

*Started: June 2026*