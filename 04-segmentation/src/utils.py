import torch
import matplotlib.pyplot as plt

def float_to_long(input: torch.Tensor) -> torch.Tensor:
  """
  input \in [0.0, ..., 1.0]
  output \in [0, 1, 2]
  """
  input = input * 255
  input = input.to(dtype=torch.long)
  input = input - 1
  return input

def close_figures():
  while len(plt.get_fignums()) > 0:
    plt.close()