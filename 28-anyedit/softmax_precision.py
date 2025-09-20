# %%

import torch
from torch.nn import functional as F

# %%

batch_size, seq_len, hidden_size = 1, 114, 4096
head_dim = 128
dtype = torch.float32

# %%

x = torch.randn((batch_size, head_dim, seq_len, seq_len), device="cuda")

# %%

att = x.to(dtype=dtype)

causal_mask = torch.tril(torch.ones((seq_len, seq_len), device="cuda", dtype=dtype))
att = att.masked_fill(causal_mask == 0, float("-inf"))

softmax_output1 = F.softmax(att, dim=-1)

att2 = att[:, :, :57, :57]
# pad_len = seq_len - 57
# if pad_len > 0:
#     att2 = F.pad(
#         att2,
#         pad=(0, pad_len, 0, pad_len),  # (dim3_left, dim3_right, dim2_left, dim2_right)
#         value=float("-inf"),
#     )
softmax_output2 = F.softmax(att2, dim=-1)

difference = (softmax_output1[0, :, 18, 18] - softmax_output2[0, :, 18, 18]).abs().sum()
print("difference:", difference)

# %%
