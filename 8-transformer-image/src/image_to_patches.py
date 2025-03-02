from torch import Tensor
from torch.nn import Module, Unfold

class ImageToPatches(Module):
  def __init__(self, patch_size: int):
    super().__init__()
    self.patch_size = patch_size
    self.unfold = Unfold(kernel_size=patch_size, stride=patch_size)

  def forward(self, x: Tensor) -> Tensor:
    """
    x: (batch_size, channels, height, width)
    y: (batch_size, num_patches, patch_size * patch_size * channels)
    """
    assert len(x.size()) == 4 # (batch_size, channels, height, width)
    # channels * height * width = patch_size * patch_size * channels * num_patches
    # num_patches = (height // patch_size) * (width // patch_size)
    y: Tensor = self.unfold(x) # (batch_size, patch_size * patch_size * channels, num_patches)
    y: Tensor = y.permute(0, 2, 1) # (batch_size, num_patches, patch_size * patch_size * channels)
    return y