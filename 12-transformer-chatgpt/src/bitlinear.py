import math
import torch
import torch.nn as nn
from torch.nn import init

class BitLinear(nn.Module):
  """
  BitNet: Scaling 1-bit Transformers for Large Language Models

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
    $\gamma = ||x||_\infty$

    $\tilde{x} = \text{Quant}(x) = \text{Clip}((x - \eta) \times \frac{Q_b}{\gamma},\epsilon,Q_b - \epsilon)$
    $$\eta = \min_{ij} x_{ij}$$
    """
    eta = x.min()
    return torch.clip(( x - eta ) * self.q_b / gamma, self.epsilon, self.q_b - self.epsilon)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    $\tilde{W} = Sign(W - \alpha)$
    $$\text{Sign}(W_{ij}) = \begin{cases} 1 & \text{if } W_{ij} \geq 0 \\ 
                                         -1 & \text{if } W_{ij} \leq 0 \end{cases}$$
    $\alpha = \frac{1}{nm} \sum_{ij} W_{ij}$

    $y = \tilde{W}\tilde{x} = \tilde{W} \, \text{Quant}(\text{LN}(x)) \times \frac{\beta\gamma}{Q_b}$
    $\text{LN}(x) = \frac{ x - E(x) }{ \sqrt{\text{Var}(x) + \epsilon} }$
    $\beta = \frac{1}{nm} ||W||_1$
    """
    tilde_w = torch.sign(self.weight - self.weight.mean())

    beta = self.weight.abs().mean()
    gamma = x.abs().max()
    tilde_x = self.quant(x=self.ln(x), gamma=gamma) * ( ( beta * gamma ) / self.q_b )
    y = tilde_x.matmul(tilde_w.T)
    return y