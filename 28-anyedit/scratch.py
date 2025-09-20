# %%

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# %%

import torch
from torch.nn import LayerNorm

ln_f = LayerNorm(4096)
x = torch.randn(2, 10, 4096)

y = ln_f(x)
print(y.shape)

# %%

from datasets import load_dataset, get_dataset_config_names

print(get_dataset_config_names("wikipedia"))

ds = load_dataset("wikipedia", "20220301.en")

# %%

def my_generator():
    print("Before first yield")
    yield 1
    print("Before second yield")
    yield 2
    print("After last yield")


def wrapped_loader():
    yield from my_generator()
    print("Wrapped loader finished")


for value in wrapped_loader():
    print("Got:", value)

# %%

import torch

A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
B = torch.tensor([9.0, 8.0])
X = torch.linalg.solve(A, B)
print(X)  # Solution to AX = B

# %%

import torch

t = torch.tensor([
    [1, 2], 
    [3, 4]
])
result = torch.gather(t, 1, torch.tensor([
    [0, 0], 
    [1, 0]
]))
print("gather result:\n", result)
# tensor([[ 1,  1],
#         [ 4,  3]])

# %%

import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
# from beartype import beartype as typechecker
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
def func(
        x: Float[Tensor, "batch"], 
        y: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch"]:
    return x + y

z = func(
    # torch.randn(1),
    torch.randn(3),
    torch.randn(3)
)
print("z shape:", z.shape)

@jaxtyped(typechecker=typechecker)
def matrix_multiply(
    x: Float[Tensor, "dim1 dim2"],
    y: Float[Tensor, "dim2 dim3"]
) -> Float[Tensor, "dim1 dim3"]:
    return x @ y

z = matrix_multiply(
    # torch.randn(1, 2, 3), 
    torch.randn(2, 3),
    torch.randn(3, 4)
)
print("z shape:", z.shape)

# %%

import torch

log_probs = torch.arange(0, 1*2*3).reshape(1,2,3)
rewriting_targets = torch.tensor([
    [-100, 1]
])
index = torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(dim=2)
loss = torch.gather(
    input=log_probs,
    dim=2,
    index=index,
)
print("log_probs:\n", log_probs) # (batch, seq, vocab)
print("rewriting_targets:\n", rewriting_targets)
print("index.shape:", index.shape)
print("index:\n", index)
print("loss:", loss)

# %%
