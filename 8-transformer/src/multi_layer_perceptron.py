from torch.nn import Sequential, Linear, GELU, Dropout

# Computation
class MultiLayerPerceptron(Sequential):
  def __init__(self, embed_size: int, dropout_probability: float):
    super().__init__(
      Linear(in_features=embed_size, out_features=embed_size * 4),
      GELU(),
      Linear(in_features=embed_size * 4, out_features=embed_size),
      Dropout(p=dropout_probability)
    )