from torch.nn import Module, Linear
from torch import Tensor

class PatchEmbedding(Module):
  def __init__(self, in_channels_size: int, out_embedding_size: int):
    super().__init__()
    # A single Layer is used to map all input patches to the output embedding dimension.
    # i.e. each image patch will share the weights of this embedding layer.
    self.embed_layer = Linear(in_features=in_channels_size, out_features=out_embedding_size)

  def forward(self, x: Tensor) -> Tensor:
    """
    x: (batch_size, num_patches, patch_size * patch_size * channels)
    y: (batch_size, num_patches, embed_size)
    """
    assert len(x.size()) == 3
    x = self.embed_layer(x)
    return x