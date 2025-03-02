from torch.nn import Module, Parameter
from src.image_to_patches import ImageToPatches
from src.patch_embedding import PatchEmbedding
from torch import randn, Tensor

class VisionTransformerInput(Module):
  def __init__(self, image_size, patch_size: int, in_channels_size: int, 
               out_embedding_size: int):
    """
    B, T, C = batch_size, num_patches, in_channels_size or out_embedding_size
    """
    super().__init__()
    self.image_to_patches_module = ImageToPatches(patch_size=patch_size)
    self.patch_embedding_module = PatchEmbedding(in_channels_size=in_channels_size,
                                                 out_embedding_size=out_embedding_size)
    # channels * height * width = patch_size * patch_size * channels * num_patches
    # num_patches = (height // patch_size) * (width // patch_size)
    num_patches = (image_size // patch_size) ** 2
    # position_embedding below is the learned embedding for the position of each patch
    # in the input image. They correspond to the cosine similarity of embeddings
    # visualized in the paper "An Image is Worth 16x16 Words"
    # https://arxiv.org/pdf/2010.11929.pdf (Figure 7, Center).
    # num_patches = (image_size // patch_size) ** 2
    self.position_embedding_module = Parameter(randn(num_patches, out_embedding_size))
  
  def forward(self, x: Tensor) -> Tensor:
    """
    x: (batch_size, channels, height, width)
    y: (batch_size, num_patches, embed_size)
    """
    x = self.image_to_patches_module(x)
    x = self.patch_embedding_module(x)
    x = x + self.position_embedding_module
    return x