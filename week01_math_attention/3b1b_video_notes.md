# 3B1B — Attention in transformers

> Note: everything described here (single-head and multi-head) is **self-attention** — the model attends to its own sequence. Cross-attention is a separate variant covered at the bottom.

- Each token is initially always the same high-dimensional vector (embedding)
- Then context from surrounding tokens could change it to a more specific vector

## Single Attention Head

- Query vector example: "As a noun, is there an adjective sitting before me?" — it's a 128-dimensional vector
- Query vector (Q) = Embedding of the token × Wq, where the same Wq matrix is used for all tokens
- Key vector is the answer to the Query vector, and it's the same dimension
- Key vector (K) = Embedding of the token × Wk, where the same Wk matrix is used for all tokens

- Wq and Wk come from training — they're learned parameters. They don't exist before training; gradient descent shapes them so that Q·K dot products become meaningful for the task.
- Q and K should be closely aligned when words are related to each other, e.g. "blue" → "creature"

- The dot product Q·K for every possible pair forms the attention pattern — higher = more relevant to updating each other's meaning
- For numerical stability, the dot product is divided by √d_k. If the K vector is 128-dimensional, you divide by √128 ≈ 11.3

> **Why divide by √d_k?**
> When you take a dot product of two high-dimensional vectors, the raw values grow roughly proportional to the dimension. A 128-dim dot product produces numbers ~11× larger than a 1-dim one, just from the math. If those numbers get too large, softmax will push one score to near 1.0 and all others near 0 — making gradients vanish during training. Dividing by √d_k keeps the scale stable regardless of dimension size.

- Since the dot products range from −∞ to +∞, softmax is applied to normalize them into weights that sum to 1

> **Why normalize with softmax?**
> You need the attention scores to become weights that sum to 1 — because you're about to use them to compute a weighted average of the Value vectors. A weighted average only makes sense if the weights sum to 1. Softmax also amplifies the largest score, so the model can "focus" strongly rather than averaging everything equally.

- Masking is sometimes applied in training so that later tokens don't influence earlier tokens

> **Why does masking matter?**
> In language modelling, the task is: "predict the next token." If token 5 can already see token 6 when computing its attention, it's cheating — it already has the answer. Masking forces the model to learn genuine prediction by blocking future positions. This is called a causal mask, and it's what makes models like GPT work at inference time: they can only attend to what came before.

- The size of the attention pattern (every K·Q combination) = context_length²

> **Why is the context² size useful to know?**
> It's a cost warning. Every token must compute a dot product with every other token — so if your context window is 1,000 tokens, the attention matrix has 1,000,000 entries. At 100,000 tokens it's 10 billion. This quadratic scaling is one of the core reasons transformer research has focused on making attention cheaper (sparse attention, linear attention, etc.). When you get to ViT in Week 6, the patch count directly controls this cost.

## Value Vectors

- Value vector (V) is what is actually used to update the embedding of each token
- V = Embedding of token × Wv
- V lives in the same high-dimensional space as the embeddings (e.g. 12,288 dimensions)

- Full attention flow per token: compute attention weights from Q·K → multiply each V by its weight → sum all weighted Vs → project through Wo → add result to the original token embedding

### Output projection (Wo)

- After computing the weighted average of V vectors, the result is in the attention head's dimension (e.g. 128), not the embedding dimension (e.g. 8)
- Wo projects the attention output back to the original embedding dimension so it can be added back to the original token

### Residual connection

- The projected attention output is added to the original embedding: `final = attention_output + original_embedding`
- The original embedding is preserved — contextual information is added on top of it, not replacing it
- This prevents vanishing gradients during backpropagation: the addition creates a shortcut that lets gradients flow directly to earlier layers without shrinking through many sequential multiplications

### Low-rank value decomposition

- Wv is not stored as a single 12,288×12,288 matrix — it is split into two smaller matrices:
  - Wv_down: 12,288×128 (projects down to a bottleneck)
  - Wv_up: 128×12,288 (projects back up to embedding space)
- Multiplied together they produce the same 12,288×12,288 shape, but at a fraction of the cost: ~3.1M parameters vs ~151M

> **Why split into two matrices (low-rank transformation)?**
> The bottleneck forces all information to pass through 128 dimensions before expanding back to 12,288. The model learns to compress — through training, those 128 dimensions come to represent the most important directions for this head's task. Everything else is discarded. This also acts as regularization, preventing the head from memorising noise. This same idea reappears in Week 7 as LoRA, which applies low-rank decomposition to fine-tuning.

## Multi-Head Attention

- Multiple attention heads run in parallel, each with their own Wq, Wk, and Wv
- Each head learns to attend to different kinds of relationships simultaneously
- The updates (weighted V sums) from all heads are added together, then added to the original embedding via the residual connection

## Why alternating attention and MLP blocks?

> One pass through attention lets each token gather information from relevant positions — but that's essentially a weighted average, which is a shallow operation. The MLP after each attention block applies a learned nonlinear transformation to that gathered information, letting the model "process" what attention collected. Stacking many attention+MLP pairs builds increasingly abstract representations: early layers detect simple relationships ("adjective before noun"), later layers handle high-level semantics ("this token refers to a specific entity in context"). You can't get that depth from a single pass.

## Cross-Attention

- Used in models that process two distinct types of data, e.g. audio + transcript, or English + French
- The only difference from self-attention: Q comes from one sequence, K and V come from the other
- This lets the model ask: "which part of sequence B is most relevant to this position in sequence A?"
- No causal masking needed — there is no notion of "future" across two separate sequences