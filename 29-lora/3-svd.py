# %%

import torch

# %%

W_0 = torch.tensor([
    [1.0, 0.0],
    [0.0, 10.0]
])

U, S, Vh = torch.linalg.svd(W_0, full_matrices=False)
print(f"{U=}")
print(f"{S=}")
print(f"{Vh=}")
I_U_1U_1_T = torch.eye(2) - U[:, 0:1] @ U[:, 0:1].t()
print(f"{I_U_1U_1_T=}")

U_2U_2_T = U[:, 1:2] @ U[:, 1:2].t()
print(f"{U_2U_2_T=}")

# %%

linear = torch.nn.Linear(2, 3, bias=False)
W = linear.weight.data
print(f"{W.shape=}")

x = torch.tensor([[1.0, 2.0]])
print(f"{x.shape=}")
y = x @ W.t()
print(f"{y.shape=}")
# %%
