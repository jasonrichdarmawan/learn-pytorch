- The query $Q$ might represent the "question" you're asking about the word (e.g., how relevant is this word to the context?).
- The key $K$ might represent the "context" or "information" available from that word.

- Suppose you have the word "king". After embedding, the query $Q_{\text{king}}$ might emphasize certain semantic features (e.g., power, royalty), while the key $K_{\text{king}}$ might emphasize a broader set of features related to the word "king" (e.g., monarchy, leadership, history).

So, for example:
- Word "Hello" (represented as $Q_1$) migt have a high similarity with "World" (represented as $K_2$) because they are related in meaning in the cotnext of your sentence, and thus the attention score between $Q_1$ and $K_2$ will be high.
- Similarly, "Hello" might have low similarity with "Hello" (the same word, but context can make the attention score different)

QK^T: How much should word i (as the query) pay attention to word j (as the key) based on how they are represented in the model.

# Simple Example with Numbers

Note: The goal is to let you know that QK^T is calculating the similarity of each word with every other word in the sequence. This important because I was imagining Q and K shape without writing it down, and thought why are we doing matrix multiplication between embedding_dimension from Q and embedding_dimension from K. In other words, for example $Q_{\text{king}}$ with $K_{\text{king}}$. Isn't that stupid? After writing it down, I was aware that matrix multiplication multiply from row of $Q$ to column of $K^T$. So, it's dot product between $Q_{\text{king}}$ and $[K_{\text{king}}, K_{\text{queen}}, K_{\text{man}}, K_{\text{woman}}]$.

From here, I concluded, it's has the property of "every output depends on the input". In other words, it's a Linear layer where $y = WX$. In this case, $w = QK^T$. And you know what? $X$ in Linear is $V$ or value in Scale Dot Product Attention. Amazing, scale dot product attention has the property of "every output depends on the input" but way less dimension (because we hyper parameter it with embed_size)

Ps: Multi-head Attention core concept is the Scale Dot Product Attention, but just focus on Scale Dot Product Attention to understand the property of "every output depends on the input", the multi-head attention is an additional concept to improve the model performance.

Consider a sequence of 3 tokens with embedding dimension 2:

```
-------> Embedding
|
|
|
v Token
Input X:
[
  [1, 2],  # Token 1
  [3, 4],  # Token 2 
  [5, 6]   # Token 3
]
```

Step 1: Compute Query, Key, Value (simplified with identity projections)

```
Q = K = V = X
```

Step 2: Compute Attention Scores (Q × K^T)

```
Attention Scores:
[
  [5,  11,  17],  # Token 1's similarity to tokens 1,2,3
  [11, 25,  39],  # Token 2's similarity to tokens 1,2,3
  [17, 39,  61]   # Token 3's similarity to tokens 1,2,3
]
```

# Multi-head Attention property of "every output depends on the input"

regarding Linear Layer == Multi-head Attention. In other words, Multi-head attention has the same property which is "every output depends on the input", but with way less weights and computation.

I found a simple way to explain it. Focus on Scaled Dot Product Attention (it's the core concept of multi head attention, multi head is just multiple scaled dot product attention).

Now, suppose we have:
Q: batch_size, sequence_length, embed_dim
K: batch_size, sequence_length, embed_dim
V: batch_size, sequence_length, embed_dim

Then,
QK^T:  batch_size, sequence_length, sequence_length

**QK^T is a matrix multiplication between every token in Q and every token in K. In other words, every output depends on every input.**

**QK^TV is another matrix multiplication. This also, every output depends on every input.**

**Suppose Linear Layer is y = WX (without bias). Then, Scaled Dot Product is W = QK^T and X = V.**

Not part of the explanation, but important: QK^TV is necessary because different tokens (in QK^TV) can have the same attention weights. So, QK^T alone is not that useful. QK^TV make it useful because even the attention pattern is the same, it retrieve different information.

Not part of the explanation, but important: Considering QK^T is a matrix multiplication ,suppose Q and K \sim N(0,1), then QK^T \sim N(0,d). Variance d is a problem because we are using softmax, and the derivative i.e. \frac{ \partial{\sigma(x_i)} }{ \partial{x_j} } is \sigma(x_i) (\delta_{ij} - \text{softmax}(x_j) ). In other words, the gradient is small when the softmax value is close to 0 or 1. Therefore, we need to do 1/\sqrt{embed_dim} to normalize