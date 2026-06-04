# Week 2 — Embeddings & Tokenization

## Tokenizers

- In practice, sentences are tokenized at the **subword level**, not word or character level
- This keeps the vocabulary manageable (~50k tokens vs millions of words), handles unseen words by breaking them into known subword pieces (e.g. "ChatGPT" → ["Chat", "G", "PT"]), and keeps sequences short enough that the attention pattern size stays reasonable
- The most common algorithm is **Byte-Pair Encoding (BPE)**: start with individual characters as tokens, then repeatedly merge the most frequent adjacent pair into a single token until the vocabulary reaches the target size
- The output of a tokenizer is a list of integers — each integer is the ID of a token in the vocabulary

## Embedding Matrix




