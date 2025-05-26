# %%

import numpy as np

def dot_product(u: list[int], v: list[int]) -> int:
    len_u = len(u)
    assert len_u == len(v), "Vectors must be of the same length"
    
    return sum(u[i] * v[i] for i in range(len_u))

u = [1, 2, 3]
v = [4, 5, 6]

assert np.dot(u, v) == dot_product(u, v), "Dot product calculation is incorrect"

# %%
