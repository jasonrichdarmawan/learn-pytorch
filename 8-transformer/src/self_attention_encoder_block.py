from torch.nn import Module, LayerNorm, MultiheadAttention
from src.multi_layer_perceptron import MultiLayerPerceptron
from torch import Tensor

class SelfAttentionEncoderBlock(Module):
  """
  This is a single self-attention encoder block, which has a multi-head attention
  block within it. The MultiHeadAttention block performs communication,
  while the MultiLayerPerceptron performs computation.
  """
  def __init__(self, embed_size: int, num_heads: int, dropout_probability: int):
    super().__init__()
    self.embed_size = embed_size
    self.layer_norm1 = LayerNorm(normalized_shape=embed_size)
    self.multi_head_attention = MultiheadAttention(embed_dim=embed_size,
                                                   num_heads=num_heads, 
                                                   dropout=dropout_probability)
    self.layer_norm2 = LayerNorm(normalized_shape=embed_size)
    self.multi_layer_perceptron = MultiLayerPerceptron(embed_size=embed_size, dropout_probability=dropout_probability)
  
  def forward(self, x: Tensor) -> Tensor:
    """
    x: (batch_size, seq_length, embed_size)
    First Layer Normalization (self.layer_norm1):
      Normalize the input.
    Multi-Head Attention (self.multi_head_attention):
      Compute the attention, and add the result back to the inptu via a residual connection.
    Second Layer Normalization (self.layer_norm2):
      Normalize again.
    MLP (self.multi_layer_perceptron):
      Apply a feed-forward neural network to transform the output.
    Final Residual Connection: Add the output of the MLP back to the input.
    """
    y = self.layer_norm1(x)
    # residual connection (x = x + ...), it helps avoid the vanishing gradient problem.
    # By adding the input to the output, the model has a shortcut path for gradients
    # to flow directly through, improving training stability.
    x = x + self.multi_head_attention(y, y, y, need_weights=False)[0]
    x = x + self.multi_layer_perceptron(self.layer_norm2(x))
    return x