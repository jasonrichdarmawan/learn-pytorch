# Reference:
# 1. [Transformer](https://www.youtube.com/watch?v=C9QSpl5nmrY)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import lightning as L

class PositionEncoding(nn.Module):
  """
  pre-compute y-axis values and store them in a matrix

  each token specified with pos,
  each embedding position specified with i.

  if each token has 4 word embeddings, then d_model is 4

  Example:
  what is statquest awesome <EOS>
  what -> pos=0
  is -> pos=1

  general form:
  1. PE(pos,2i) = sin(pos/10000^(2i/d_model))
  2. PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
  """
  def __init__(self, d_model=2, max_len=6):
    super().__init__()
    
    pe = torch.zeros(max_len, d_model)
    # pe: (max_len, d_model)

    position = torch.arange(start=0, end=max_len, step=1, 
                            dtype=torch.float32).unsqueeze(1)
    # position: (max_len, 1)
    embedding_index = torch.arange(start=0, end=d_model, step=2, 
                                   dtype=torch.float32)
    # embedidng_index represents the index, i
    # step=2 for optimization purpose of 2i/d_model
    # PE(pos,2i) = sin(pos/10000^(2i/d_model))
    # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

    div_term = 1 / torch.tensor(10000, dtype=torch.float32) ** (embedding_index / d_model)

    # sin, cos, sin, cos, ...
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe)
  
  def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Args:
    word_embeddings: (n, token_len, d_model]
    """
    word_embeddings_len = word_embeddings.size(1)
    return word_embeddings + self.pe[:word_embeddings_len]

class Attention(nn.Module):
  def __init__(self, d_model=2):
    super().__init__()
    self.d_model = d_model

    self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
  
  def forward(self, encondings_for_q: torch.Tensor,
              encondings_for_k: torch.Tensor, encondings_for_v: torch.Tensor,
              mask: torch.Tensor=None) -> torch.Tensor:
    """
    Args:
    encondings_for_q: (n, q_len, d_model)
    encondings_for_k: (n, k_len, d_model)
    encondings_for_v: (n, v_len, d_model)

    Returns:
    torch.Tensor: (n, q_len, d_model)
    """
    q: torch.Tensor = self.W_q(encondings_for_q) # (n, q_len, d_model)
    k: torch.Tensor = self.W_k(encondings_for_k) # (n, k_len, d_model)
    v: torch.Tensor = self.W_v(encondings_for_v) # (n, v_len, d_model)

    # scaled dot-product attention
    # Q: (n, q_len, d_model)
    # K: (n, k_len, d_model)
    # K^T: (n, d_model, k_len)
    # Q * K^T: (n, q_len, k_len)
    qk_t: torch.Tensor = torch.einsum('nqd,nkd->nqk', q, k)

    if mask is not None:
      qk_t = qk_t.masked_fill(mask=mask, value=float('-inf'))
      # Q * K^T + M
      # M: (token_len, token_len)

    qk_t = qk_t / (self.d_model ** 0.5)
    # ( Q * K^T + M ) / sqrt(d_model)

    qk_t = F.softmax(qk_t, dim=-1)
    # normalize weights over the key_len

    qk_t_v = torch.einsum('nql,nld->nqd', qk_t, v)
    # k_len == q_len
    # qk_t: (n, q_len, k_len)
    # value: (n, v_len, d_model)
    # QK^T * V: (n, q_len, d_model)
    return qk_t_v
  
class DecoderOnlyTransformer(L.LightningModule):
  def __init__(self, num_tokens=4, d_model=2, max_len=6):
    """
    Args:
    num_tokens: number of tokens in the vocabulary
    d_model: dimension of the word embeddings
    max_len: maximum length of the input sequence. for position encoding
    """
    super().__init__()

    self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
    self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
    self.self_attention = Attention(d_model=d_model)
    self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Args:
    token_ids: (n, token_len)

    Returns:
    torch.Tensor: (n, token_len, num_tokens)
    """
    word_embeddings: torch.Tensor = self.we(token_ids)
    # word_embeddings: (n, token_len, d_model)
    position_encoded: torch.Tensor = self.pe(word_embeddings)
    # position_encoded: (n, token_len, d_model)

    token_len = token_ids.size(1)
    mask = torch.tril(torch.ones(token_len, token_len, device=self.device)) == 0

    self_attention_values = self.self_attention(encondings_for_q=position_encoded,
                                                encondings_for_k=position_encoded,
                                                encondings_for_v=position_encoded,
                                                mask=mask)
    # self_attention_values: (n, q_len, d_model)
    
    residual_connection_values = position_encoded + self_attention_values
    # residual_connection_values: (n, q_len, d_model)

    fc_layer_output = self.fc_layer(residual_connection_values)
    # fc_layer_output: (n, q_len, num_tokens)

    return fc_layer_output
  
  def configure_optimizers(self):
    return Adam(self.parameters(), lr=0.1)
  
  def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
    """
    Args:
    batch: (input_tokens, labels)
    """
    input_tokens, labels = batch
    # input_tokens: (n, token_len)
    # labels: (n, token_len)
    output = self.forward(input_tokens)
    # output: (n, q_len, num_tokens)
    output = output.transpose(1,2)
    # output: (n, num_tokens, q_len)
    loss = self.criterion(output, labels)
    # q_len == token_len
    # expected output: (n, num_tokens, q_len)
    # expected labels: (n, token_len)
    return loss