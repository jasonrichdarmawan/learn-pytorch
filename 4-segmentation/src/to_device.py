from torch import Tensor
from torch.nn import Module
from torch.cuda import is_available

def get_device() -> str:
  return 'cuda' if is_available() else 'cpu'

class ToDevice:
  def __init__(self, device):
    self.device = device
  
  def __call__(self, x: Tensor) -> Tensor:
    return x.to(self.device)