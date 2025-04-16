# %%

import torch

W = torch.arange(0, 12).reshape(3, 4)
x = torch.arange(0, 4).reshape(4, 1) # column vector
print(f"W\n{W}")
print(f"x\n{x}")
print(f"W^T W\n{W.T @ W}")
print(f"W^T Wx\n{W.T @ W @ x}")
# example 3.1.5
# https://math.libretexts.org/Bookshelves/Linear_Algebra/Fundamentals_of_Matrix_Algebra_(Hartman)/03%3A_Operations_on_Matrices/3.01%3A_The_Matrix_Transpose
# (AB)^T = B^T A^T
print(f"x.T (W^T W).T\n{(x.T @ (W.T @ W).T).T}")
