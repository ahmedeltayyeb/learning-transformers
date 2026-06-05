# Week 2 — Embeddings & Tokenization

## Tokenizers

- In practice, sentences are tokenized at the **subword level**, not word or character level
- This keeps the vocabulary manageable (~50k tokens vs millions of words), handles unseen words by breaking them into known subword pieces (e.g. "ChatGPT" → ["Chat", "G", "PT"]), and keeps sequences short enough that the attention pattern size stays reasonable
- The most common algorithm is **Byte-Pair Encoding (BPE)**: start with individual characters as tokens, then repeatedly merge the most frequent adjacent pair into a single token until the vocabulary reaches the target size
- The output of a tokenizer is a list of integers — each integer is the ID of a token in the vocabulary

## Embedding Matrix

- Shape: `(vocab_size, d_model)` — one row per token in the vocabulary
- To get a token's embedding, use its integer ID as a row index into the matrix — no matrix multiplication, just a lookup
- Values start random and are learned during training via backpropagation — over millions of examples, embeddings for semantically similar words converge to similar vectors as an emergent property of predicting text well

## Positional Encoding

- Needed because attention is order-blind — Q·K dot products have no notion of whether a token is at position 2 or position 200
- A positional encoding vector is computed for each position and **added to the token embedding** — the transformer sees meaning and position combined in a single vector
- Uses sine and cosine at different frequencies across dimensions: each dimension pair captures position at a different "zoom level", and together they create a unique fingerprint for every position
- Sine and cosine keep all values bounded between -1 and 1 — the positional signal never dominates the meaning signal regardless of sequence length
- Formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`