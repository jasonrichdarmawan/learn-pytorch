from torch import cat, split
from torchvision.datasets import OxfordIIITPet

class OxfordIIITPetAugmented(OxfordIIITPet):
  def __init__(self, root: str, split: str, target_types: str, 
                download: bool, pre_transform=None, post_transform=None,
                pre_target_transform=None, post_target_transform=None,
                common_transform=None):
    super().__init__(
        root=root,
        split=split,
        target_types=target_types,
        download=download,
        transform=pre_transform,
        target_transform=pre_target_transform
    )
    self.post_transform = post_transform
    self.post_target_transform = post_target_transform
    self.common_transform = common_transform

  def __getitem__(self, index: int):
    (input, mask) = super().__getitem__(index)

    if self.common_transform:
      both = cat([input, mask], dim=0)
      both = self.common_transform(both)
      input, mask = split(both, split_size_or_sections=3, dim=0)
    
    if self.post_transform:
      input = self.post_transform(input)
    if self.post_target_transform:
      mask = self.post_target_transform(mask)

    return input, mask