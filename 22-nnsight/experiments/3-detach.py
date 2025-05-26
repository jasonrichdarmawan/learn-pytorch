# %%

import torch

input = torch.tensor([1.0, 2.0], requires_grad=True)
loss = input.sum()
loss.backward()
print(f"loss: {loss}")
print(f"input.grad: {input.grad}")
print(f"input.grad[0]: {input.grad[0]}") # ∂loss/∂input[0] = d(input[0] + input[1])/∂input[0] = 1
print(f"input.grad[1]: {input.grad[1]}") # ∂loss/∂input[1] = d(input[0] + input[1])/∂input[1] = 1

# %%

import torch

input = torch.tensor([1.0, 2.0])
weight = torch.tensor([2.0, 3.0], requires_grad=True)

loss = (input * weight).sum()
loss.backward()
print(f"loss: {loss}")
print(f"weight.grad: {weight.grad}")
print(f"weight.grad[0]: {weight.grad[0]}") # ∂loss/∂weight[0] = d(input[0] * weight[0] + input[1] * weight[1])/∂weight[0] = a[0]
print(f"weight.grad[1]: {weight.grad[1]}") # ∂loss/∂weight[1] = d(input[0] * weight[0] + input[1] * weight[1])/∂weight[1] = a[1]

# %%

import torch
import torch.nn.functional as F

input = torch.tensor([1.0, 2.0])
weight = torch.tensor([2.0, 3.0], requires_grad=True)

logits = input * weight

loss = (logits - logits).sum()
loss.backward()
print(f"logits: {logits}")
print(f"loss: {loss}")
print(f"weight.grad: {weight.grad}")
print(f"weight.grad[0]: {weight.grad[0]}")
print(f"weight.grad[1]: {weight.grad[0]}")

# %%
"""
Let's break down why the gradients differ in these two scenarios.
The key is understanding how `.detach()` affects
the computation graph used for backpropagation.

1. `loss = (logits - logits).sum()`
   - `logits = input * weight`. Since `weight` has 
     `requires_grad=True`, `logits` is part of the computation
     graph and depends on `weight`.

   - The expression `logits - logits` involves the same `logits`
     tensor twice. Both instances are connected to `weight`
     in the computation graph.
   - Mathematically, `logits - logits` is always zero.
   - When you call `loss.backward()`, PyTorch calculates
     the gradient of `loss` with respect to `weight` using the
     chain rule: `∂loss/∂weight = ∂loss/∂logits * ∂logits/∂weight`
   - `∂logits/∂weight`` is simply `input` (`[1.0, 2.0]`)
   - `∂loss/∂logits` is the gradient of 
     `(logits[0] - logits[0] + logits[1] - logits[1])` with
     respect to `logits`. Since `logits` appears positively
     and negatively, the derivatives cancel out. The derivative of
     `(y - y)` with respect to `y` is `1 - 1 = 0`.
   - So, `∂loss/∂logits = [0., 0.0]`
   - Therefore, 
     `∂loss/∂weight = [0., 0.] * [1.0, 2.0] = [0.0, 0.0]`
2. `loss = (logits.detach() - logits).sum()`
    - `logits = input * weight` is the same as before.
      `logits` depends on `weight`.
    - `logits.detach()` creates a new tensor that share the
      same data as `logits` but is **detached** from the
      computation graph. It's treated as constant during
      backpropagation; gradients will not flow back through it.
    - The expression is now `(constant_tensor - logits).sum()`.
      Only the second `logits` term is connected to `weight`
      in the graph used for gradient calculation.
    - When you call `loss.backward()`, PyTorch calculates
      `∂loss/∂weight = ∂loss/∂logits * ∂logits/∂weight`.
    - `∂logits/∂weight` is still `input` (`[1.0, 2.0]`).
    - `∂loss/∂logits` is the gradient of
      `(logits_detached[0] - logits[0] + 
        logits_detached[1] - logits[1])` with respect to `logits`.
      Since `logits_detached` is constant, the derivative is
      effectively `∂(-logits[0] - logits[1]) / ∂logits`.
    - The partial derivative with respect to `logits[0]` is -1.
    - The partial derivative with respect to `logits[1]` is -1.
    - So, `∂loss/∂logits = [-1., -1.]`
    - Therefore, `∂loss/∂weight = [-1., -1.] * [1.0, 2.0]
      = [-1.0, -2.0]`. The gradient `weight.grad` will be 
      `[-1., -2]`
"""

import torch
import torch.nn.functional as F

input = torch.tensor([1.0, 2.0])
weight = torch.tensor([2.0, 3.0], requires_grad=True)

logits = input * weight

loss = (logits.detach() - logits).sum()
loss.backward()
print(f"logits: {logits}")
print(f"loss: {loss}")
print(f"weight.grad: {weight.grad}")
print(f"weight.grad[0]: {weight.grad[0]}")
print(f"weight.grad[1]: {weight.grad[0]}")

# %%

import torch

x = torch.tensor([2.0], requires_grad=True)

y = x * 3 # dy/dx = 3

z = y * y # dz/dy = 2*y = 2*(3x) = 6x = 12

loss = z

loss.backward()

"""
Check the gradient of x
Chain rule: d(loss)/dx = d(loss)/dz * dz/dy * dy/dx
d(loss)/dz = 1 (gradient of z w.r.t. itself)
dz/dy = 2*y = 2*(3*2.0) = 12
dy/dx = 3
d(loss)/dx = 1 * 12.0 * 3 = 36.0
"""
print(f"x.grad: {x.grad}")
"""
Here, both `y = x * 3` and `z = y * y` are part of the graph
connecting `loss` back to `x`. The gradient `x.grad` reflects
the derivative of the entire chain of calculations (`z` with
respect to `x`)
"""

# %%

import torch

x = torch.tensor([2.0], requires_grad=True)

y = x * 3

y_detached = y.detach()

z = y_detached * y_detached # dz / dy_detached = 2 * y_detached = 12, but no path back to x

loss = z

"""
Backpropagate
This will raise an error because `loss` doesn't depend on any 
tensor requiring gradients in the graph. `y_detached` broke
the chain back to `x`
"""
try:
  loss.backward()
except RuntimeError as e:
  print(f"Error: {e}")
  print(f"x.grad: {x.grad}")

# %%

import torch

x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = x * 2
loss = y.detach() + z # d(loss)/dx = d(y.detach())/dx + dz/dx = 0 + 2 = 2
loss.backward()
print(f"x.grad: {x.grad}")

"""
In the second part, when we calculate `loss = y.detach() + z`,
the calculation involving `y.detach()` does not contribute
to the gradient of `x` because the connection was severed by
`.detach()`. Only the `z = x * 2` part contributes, resulting in
`x.grad` being 2.

In short, any differentiable PyTorch operation involving a tensor
that requires gradients becomes part of the computation graph
and influence the gradients calculated by `.backward()`.
Using `.detach()` (or operating within with `with torch.no_grad():`)
prevents subsequent operation from being added to the graph for
that specific part
"""

# %%

x = torch.tensor([2.0])
y = x
y += torch.tensor([2.0])

print(f"x: {x}")
print(f"y: {y}")

# %%

x = torch.tensor([2.0])
y = x.detach()
y += torch.tensor([2.0])

print(f"x: {x}")
print(f"y: {y}")

# %%

x = torch.tensor([2.0])
y = x.clone()
y += torch.tensor([2.0])

print(f"x: {x}")
print(f"y: {y}")

# %%

x = torch.tensor([2.0], requires_grad=True)
y = 2 * x
y.backward()

grad = x.grad
grad += 1

y = 2 * x
y.backward()

print(f"x: {x}")
print(f"y: {y}")
print(f"x.grad: {x.grad}")
print(f"grad: {grad}")

# %%

x = torch.tensor([2.0], requires_grad=True)
y = 2 * x
y.backward()

grad = x.grad.detach()
grad += 1

y = 2 * x # dy/dx = 2
y.backward()

print(f"x: {x}")
print(f"y: {y}")
print(f"x.grad: {x.grad}")
print(f"grad: {grad}")

# %%