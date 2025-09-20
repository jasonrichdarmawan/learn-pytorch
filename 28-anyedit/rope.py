# %%

import torch

# 4 tokens, 2-dim embeddings
embeddings = torch.tensor([
    [1.0, 0.0],  # token 0
    [0.0, 1.0],  # token 1
    [1.0, 1.0],  # token 2
    [0.5, -0.5], # token 3
])  # shape: (4, 2)

# Position IDs for each token
position_ids = torch.arange(4)  # [0, 1, 2, 3]

# %%

import math

def get_freqs(position_ids):
    # For 2D, use a single frequency
    theta = position_ids * (math.pi / 2)  # [0, π/2, π, 3π/2]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin

cos, sin = get_freqs(position_ids)
print("cos:", cos)
print("sin:", sin)

# %%

def rotate_half(x):
    # For 2D: [-x2, x1]
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    # x: (seq_len, 2)
    # cos/sin: (seq_len,)
    # Apply element-wise
    return x * cos.unsqueeze(-1) + rotate_half(x) * sin.unsqueeze(-1)

rope_embeddings = apply_rope(embeddings, cos, sin)
print("RoPE embeddings:\n", rope_embeddings)

# %%

# Longer sequence (6 tokens)
embeddings_long = torch.tensor([
    [1.0, 0.0],  # token 0
    [0.0, 1.0],  # token 1
    [1.0, 1.0],  # token 2
    [0.5, -0.5], # token 3
    [2.0, 2.0],  # token 4
    [3.0, 3.0],  # token 5
])
position_ids_long = torch.arange(6)

cos_long, sin_long = get_freqs(position_ids_long)
rope_embeddings_long = apply_rope(embeddings_long, cos_long, sin_long)

# Slice to first 4 tokens
rope_embeddings_long_sliced = rope_embeddings_long[:4]
print("RoPE embeddings (long, sliced):\n", rope_embeddings_long_sliced)

# %%