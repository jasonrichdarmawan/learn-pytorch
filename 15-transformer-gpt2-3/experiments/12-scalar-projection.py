# %%

import torch
import matplotlib.pyplot as plt

# Define two vectors
a = torch.tensor([3, 4], dtype=torch.float32)
b = torch.tensor([4, 3], dtype=torch.float32)
print(f"b magnitude: {torch.norm(b)}")

# Compute dot product and scalar projection
dot_product = torch.dot(a, b)
print(f"dot product: {dot_product}")
scalar_projection = dot_product / torch.norm(b)

# Visualization
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='r', label='a')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='b', label='b')

# Project v1 onto v2
proj_v1_on_v2 = (dot_product / torch.norm(b)**2) * b
plt.quiver(0, 0, proj_v1_on_v2[0], proj_v1_on_v2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Vector projection of a on b')

# Set plot limits and labels
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title(f"Scalar Projection: {scalar_projection:.2f}")
plt.show()
