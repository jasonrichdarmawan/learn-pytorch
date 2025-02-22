from torch.nn import Sequential, Linear, GELU, Dropout
from torch import Tensor

class MultiLayerPerceptron(Sequential):
  """
  An MLP is a type of neural network composed of layers of fully connected neurons (also known as dense layers)
  Dimensional Expansion and Compression: 
    The first linear layer expands the feature space (embed_size \to embed_size * 4).
    This allows the model to learn more complex and expressive representations of the input.
    Afterward, the second linear layer reduces it back to the original embedding size,
    allowing the network to adjust the learned features.
  Non-Linearity with GELU: 
    The GELU activation adds non-linearity, which is crucial for learning complex patterns.
    Without activation functions, the network would essentially become just a linear transformation,
    limiting its ability to model complex data distributions.
  Regularization with Dropout:
    Dropout helps prevent overfitting, especially in large networks,
    by ensuring that the model does not rely to oheavily on any single unit.
    This is particularly important in deep learning models,
    where overfitting to the training data can hurt performance on unseen data.
  """
  def __init__(self, embed_size: int, dropout_probability: float):
    super().__init__()
    self.layers = Sequential(
      Linear(in_features=embed_size, out_features=embed_size * 4),
      GELU(),
      Linear(in_features=embed_size * 4, out_features=embed_size),
      Dropout(p=dropout_probability),
    )
  
  def forward(self, x: Tensor) -> Tensor:
    return self.layers(x)