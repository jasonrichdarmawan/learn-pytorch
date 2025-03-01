# Reference:
# 1. [PyTorch Transformer from Scratch](https://www.youtube.com/watch?v=U0s0f995w14)
# 2. [SelfAttention vs Multi-head Self Attention](https://www.youtube.com/shorts/Muvjex0nkes)

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  """
  This is multi-head self-attention mechanism
  Other name: Scaled Dot Product Attention
  """
  def __init__(self, embed_size: int, heads: int):
    """
    if we have embed_size of 256 and heads of 8, then each head will have 32 features
    """
    super().__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

    self.linear_value = nn.Linear(self.embed_size, self.embed_size, bias=False) # W_v^(1), ..., W_v^(h)
    self.linear_key = nn.Linear(self.embed_size, self.embed_size, bias=False) # W_k^(1), ..., W_k^(h)
    self.linear_query = nn.Linear(self.embed_size, self.embed_size, bias=False) # W_q^(1), ..., W_q^(h)
    self.linear_out = nn.Linear(self.embed_size, embed_size) # fully connected layer

  def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, 
              mask: torch.Tensor) -> torch.Tensor:
    """
    Each token is represented by a vector with shape embed_size.
    - The multi-dimensional space allows splitting attention into multiple heads
    - Each head can focus on different aspects of the data
    - Some heads might focus on syntatic relationships, others on semantic ones.

    Consider encoding the word "bank":
    - 1D encoding: Maybe a single number like 42
      - Cannot distinguish between financial institution vs. river bank
    - 256 encdoing: A rich vector where different dimensions represent:
      - Relation to finance
      - Relation to nature
      - Formality level
      - Part of speech
      - And hundreds more subtle properties

    Original embedding: (N, seq_length, embed_size) or x_i \in \mathbb{R}^{seq_length \times embed_size}
    Split into self.heads: (N, seq_length, heads, head_dim) where heads * head_dim == embed_size
    Process attention separately for each head
    Recombine: (N, seq_length, heads, head_dim) -> (N, seq_length, embed_size)
    """
    N = query.shape[0]
    value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

    value = self.linear_value(value) # shape: (N, value_len, embed_size)
    key = self.linear_key(key)
    query = self.linear_query(query)

    # # Split the embedding into self.heads pieces
    value = value.view(N, value_len, self.heads, self.head_dim)
    key = key.view(N, key_len, self.heads, self.head_dim)
    query = query.view(N, query_len, self.heads, self.head_dim)

    energy = torch.einsum("nqhd,nkhd->nhqk", query, key) # sum over the head_dim
    # energy or (QK^T)_{T\times T}
    # queries shape: (N, query_len, heads, head_dim)
    # keys shape: (N, key_len, heads, head_dim)
    # energy shape: (N, heads, query_len, key_len)

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))
    
    attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) # normalize over the key_len
    # when computing dot products between query and key vectors,
    # the magnitude grows with the embedding dimensions d. Without scaling:
    # 1. Preventing Extremely Small Gradients
    # - As d increases, the dot products become larger in magnitude
    # - After applying softmax, this pushes the distribution toward extremely peaked values (near 0 or 1)
    # - These extremely peaked softmax values have very small gradients
    #   - This is because the gradient of the softmax function with respect to its inputs has the form:
    #     - \frac{ \partial{\sigma(x_i)} }{ \partial{x_j} } = \text{softmax}(x_i) (\delta_{ij} - \text{softmax}(x_j)
    #     - The gradient is small when the softmax value is close to 0 or 1
    #     - Note: \frac{ \partial{ \sigma(x_i) }{ \partial{x_j} } tells us how the softmax output changes as z_k changes.
    # - Small gradients lead to slow or unstable training
    # 2. Maintaining Appropriate Variance
    # From a statistical perspective:
    # - If query and key vectors have components with mean 0 and variance 1
    # - The dot product of two such vectors will have variance d (where d is the embedding dimension)
    #   - q = [q1, q2, ..., qd], k = [k1, k2, ..., kd]
    #   - q \cdot k = q1 * k1 + q2 * k2 + ... + qd * kd
    #   - E[q \cdot k] = E[q1 * k1 + q2 * k2 + ... + qd * kd] = E[q1 * k1] + E[q2 * k2] + ... + E[qd * kd] = 0
    #   - Var[q \cdot k] = Var[q1 * k1 + q2 * k2 + ... + qd * kd]
    #                    = Var[q1 * k1] + Var[q2 * k2] + ... + Var[qd * kd]
    #     - Variance of product of two random variable
    #     - Var[q1 * k1] = E[q1]^2.Var(k_1) + E[k1]^2.Var(q1) + Var(q1).Var(k1)
    #                    = 0 + 0 + 1 * 1 = 1
    #     - Var[q \cdot k] = 1 + 1 + ... + 1 (d times)
    #                      = d

    # - Scaling by 1/\sqrt{d} normalizes this variance back to 1
    # - This keeps the attention weights in a reasonable range regardless of embedding size

    out = torch.einsum("nhql,nlhd->nqhd", attention, value).reshape(N, query_len, self.heads * self.head_dim)
    # key_len and value_len are the same
    # attention shape: (N, heads, query_len, key_len)
    # values shape: (N, value_len, heads, head_dim)
    # after einsum shape: (N, query_len, heads, head_dim) then flatten last two dimensions

    out = self.linear_out(out)
    return out
  
class TransformerBlock(nn.Module):
  def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int):
    super().__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embed_size, embed_size)
    )

    self.dropout = nn.Dropout(dropout)
  
  def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
      ---> Add & Norm
      |       ^
      |       |
      | Feed Forward
      |       ^
      --------|
              |
      ---> Add & Norm
      |        ^
      |        |
      | Multi-Head Attention
      |      ^  ^   ^
      |      |  |  |
      |      -------
      |         |
      ----------|
    """
    attention = self.attention(value, key, query, mask)
    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))
    return out
  
class Encoder(nn.Module):
  def __init__(self, src_vocab_size: int, embed_size: int, num_layers: int, heads: int, 
                device: str, forward_expansion: int, dropout: float, max_length: int):
    super().__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion
        )
        for _ in range(num_layers)
      ]
    )

    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
    out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

    for layer in self.layers:
      out = layer(out, out, out, mask) # value, key, query, mask
    
    return out
    
class DecoderBlock(nn.Module):
  def __init__(self, embed_size: int, heads: int, forward_expansion: int, dropout: int, 
               device: str):
    super().__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(embed_size, heads, dropout, 
                                              forward_expansion)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.Tensor, value: torch.Tensor, key: torch.Tensor, 
              src_mask: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
    attention = self.attention(x, x, x, trg_mask) # Masked Multi-Head Attention (processes target sequence)
    query = self.dropout(self.norm(attention + x)) # Add & Norm
    # value and key are the encoder output
    # querying the encoder's output (keys and values) to gather relevant information
    # the naming reflects its role in the next attention mechanism
    out = self.transformer_block(value, key, query, src_mask) # Cross-Attention (attends to encoder output) + FFN (processes each position individually)
    return out
  
class Decoder(nn.Module):
  def __init__(self, trg_vocab_size: int, embed_size: int, num_layers: int, heads: int,
               forward_expansion: int, dropout: float, device: str, max_length: int):
    super().__init__()
    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)
    self.layers = nn.ModuleList(
      [
        DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
        for _ in range(num_layers)
      ]
    )
    self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, 
              trg_mask: torch.Tensor) -> torch.Tensor:
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
    x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

    for layer in self.layers:
      x = layer(x, enc_out, enc_out, src_mask, trg_mask) # x, value, key, src_mask, trg_mask
    
    out = self.fc_out(x)
    return out
  
class Transformer(nn.Module):
  def __init__(self, src_vocab_size: int, trg_vocab_size: int, src_pad_idx: int,
               trg_pad_idx: int, embed_size: int = 256, num_layers: int = 6,
               forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
              device: str = "cuda", max_length: int = 100):
    super().__init__()
    self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device,
                            forward_expansion, dropout, max_length)
    self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads,
                            forward_expansion, dropout, device, max_length)
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device
  
  def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    # (N, 1, 1, src_len)
    return src_mask.to(self.device)
  
  def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
    return trg_mask.to(self.device)
  
  def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)
    enc_src = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_src, src_mask, trg_mask)
    return out
  
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], 
                    [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
  trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                      [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
  
  src_pad_idx = 0
  # Purpose of padding:
  # - Transformers need fixed-length inputs for batching
  # - Shorter sequences are padded to match the length of the longest sequence in a batch.
  # Masking Mechanism:
  # - The `make_src_mask` function creates a mask to ignore padding tokens during attention
  # - This prevents the model from attending to or learning from meaningless padding tokens
  # - The mask essentially tells the attention mechanism:
  #   "don't look at positions where tokens are padding"
  trg_pad_idx = 0
  src_vocab_size = 10
  trg_vocab_size = 10
  # src_vocab_size = 10 and trg_vocab_size = 10 set the size of the vocabular for the source and target sequences:
  # 1. Token Range:
  #    - Tokens are represented as integeres from 0 to vocab_size - 1
  #    - That's why no value in x or trg exceeds 9 (with vocab size 10)
  # 2. Embedding Layer:
  #    - The nn.Embedding(src_vocab_size, embed_size) creates a lookup table for 10 unique tokens
  #    - Each token (0-9) is mapped to an embedding vector of size embed_size
  #    - If you tried to use a token >= 10, you'd get an out-of-bounds error.
  # 3. Real-world usage:
  #    - In practice, vocabulary sizes are much larger (thousands to tens of thousands)
  model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
  out = model(x, trg[:,:-1])
  # During training, the transformer decoder takes the target sequence 
  # shifted to the right as input. It then predicts the next token for each position.
  # When training, the expected outputs are trg[:,1:] (target shifted left),
  # while inputs are trg[:,:-1] (target shifted right).
  print(trg[:,:-1])
  print(out.shape)