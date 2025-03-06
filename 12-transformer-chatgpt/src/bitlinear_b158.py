import math
import torch
import torch.nn as nn
from torch.nn import init

class BitLinearb158(nn.Module):
  """
  BitNet: Scaling 1-bit Transformers for Large Language Models

  The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

  Disclaimer:
    Maybe the implementation is not correct.
  """
  def __init__(self, in_features: int, out_features: int, bias=False, b: int = 8, epsilon: float = 1e-5) -> None:
    super().__init__()

    self.weight = nn.Parameter(torch.empty(out_features, in_features))

    # $\text{LN}(x) = \frac{ x - E(x) }{ \sqrt{\text{Var}(x) + \epsilon} }$
    self.ln = nn.LayerNorm(normalized_shape=in_features, elementwise_affine=False, bias=False)

    # $Q_b = 2^{b-1}$
    self.q_b = 2**(b-1)
    self.epsilon = epsilon

    self.reset_parameters()
  
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
  
  def quant(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    $\tilde{x} = \text{Quant}(x) = \text{Clip}(x \times \frac{Q_b}{\gamma}, -Q_b + \epsilon, Q_b - \epsilon)$
    $\text{Clip}(x,a,b) = \max(a, \min(b,x))$
    $\gamma = ||x||_\infty$
    """
    return torch.clip(x * self.q_b / gamma, -self.q_b + self.epsilon, self.q_b - self.epsilon)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    $\tilde{W} = \text{RoundClip(\frac{W / \gamma + \epsilon}, -1, 1)}$
    $\text{RoundClip}(x,a,b) = max(a, min(b, round(x)))$

    $y = \tilde{W}\tilde{x} = \tilde{W} \, \text{Quant}(\text{LN}(x)) \times \frac{\beta\gamma}{Q_b}$
    $\text{LN}(x) = \frac{ x - E(x) }{ \sqrt{\text{Var}(x) + \epsilon} }$
    $\beta = \frac{1}{nm} ||W||_1$
    """
    w_gamma = self.weight.abs().mean()
    tilde_w = torch.clip(torch.round(self.weight / (w_gamma + self.epsilon)), -1, 1)

    beta = self.weight.abs().mean()
    x_gamma = x.abs().max()
    tilde_x = self.quant(x=self.ln(x), gamma=x_gamma) * ( ( beta * x_gamma ) / self.q_b )
    y = tilde_x.matmul(tilde_w.T)
    return y