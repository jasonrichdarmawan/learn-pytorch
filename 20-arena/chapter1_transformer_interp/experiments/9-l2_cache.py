# %% [markdown]

"""
The difference in performance between computing
the full matrix multiplication `W_E @ W_V @ W_O @ W_U`
at once versus chunking it with `W_E[indices] @ W_V @ W_O @ W_U`
is primarily due to how modern processors, especially
GPUs, utilize their memory hierachy (like L1 and L2 caches).

In essence, the hierarchy is: Processor Core -> L1 Cache -> L2 Cache -> Main Memory (VRAM).
The closer the data is to the core, the faster it can be accessed.
The goal of performance optimization techniques like the chunknig in your
script is to keep the working data set small enough to fit into
the faster L1 and L2 caches as much as possible,
avoiding the performance penalty of fetching data from VRAM.

An RTX 4090 has 128 SMs. So:
- The L1 cache avaialble to a single SM is at most 128 KB.
- The shared L2 cache avaialble to that same SM
  (and all others) is 72 MB.

Here's a breakdown of why chunking can be fasteR:
1. Cache Utilization: GPUs have a very fast but
   small caches (L1, L2). When you perform a massive
   multiplication like `W_E @ W_V @ W_O @ W_U`,
   the intermediate matrices might be too large
   to fit into these caches. For example, the result
   of `W_E @ W_V @ W_O` is `(vocab_size, d_model)`
   matrix, which is about 77 MB (assumming
   `vocab_size` is 50304 and `d_model` is 768,
   with float16 dtype). This is likely larger than
   your GPU's L2 cache. When data doesn't fit in the cache,
   it must be fetched from the much slower main GPU
   memory (VRAM), causing significant delays
2. Chunking and Caching: When you process
`W_E` in smaller chunks (e.g., by setting
`split_size` to a value smaller than `vocab_size`),
you are performing a series of smaller matrix
multiplications.
- The intermediate matrices are smaller. For instance,
  if `split_size` is 1024, the result of `W_E[indices] @ W_V @ W_O`
  is `(split_size, d_model)`, which is only ~3 MB 
  (assuming d_model is 768, dtype is float32).
- This smaller intermediate result, along with
  the required rows of `W_V` and `W_O`, can
  more easily fit into the L2 cache.
- When the final multiplication with `W_U` happens,
  the necessary data is already in the fast cache,
  leading to a much faster computation for that chunk.

By iterating through chunks of `W_E`, you allow
the GPU to reuse the data for `W_V`, `W_O`, and
`W_U` that is already in its fast cache,
minimizing slow memory accesses to VRAM. Even though you
are adding loop overhead, the time saved by
avoiding VRAM latency is far greater,
resulting in a faster overall computation time.
"""

# %%

import torch
from time import perf_counter

device = "cuda:0"
dtype = torch.float32

vocab_size = 50304
d_model = 768
n_heads = 12
d_head = d_model // n_heads

iterations = 10

split_size = 1024

W_E = torch.arange(
  start=0, 
  end=vocab_size * d_model, 
  dtype=dtype,
  device=device,
)
W_E = W_E.reshape(vocab_size, d_model)

W_V = torch.arange(
  start=0,
  end=d_model * d_head * n_heads, 
  dtype=dtype,
  device=device,
)
W_V = W_V.reshape(d_model, n_heads * d_head)

W_O = torch.arange(
  start=0, 
  end=d_model * d_model, 
  dtype=dtype,
  device=device,
)
W_O = W_O.reshape(d_model, d_model)

indices = torch.arange(start=0, end=split_size, device=device)
W_E_W_V_W_O = W_E[indices] @ W_V @ W_O
print(f"Size of W_E[split_size] @ W_V @ W_O in KB: {W_E_W_V_W_O.nelement() * W_E_W_V_W_O.element_size() / 1024:.2f}")

W_U = torch.arange(
  start=0, 
  end=d_model * vocab_size, 
  dtype=dtype,
  device=device,
)
W_U = W_U.reshape(d_model, vocab_size)

elapsed_times = []

for i in range(iterations):
  torch.cuda.empty_cache()

  start_time = perf_counter()
  for indices in torch.split(
    tensor=torch.arange(start=0, end=vocab_size, device=device), 
    split_size_or_sections=split_size,
  ):
    _ = W_E[indices] @ W_V @ W_O @ W_U

  torch.cuda.synchronize(device)
  end_time = perf_counter()
  elapsed_times.append(end_time - start_time)

  del _

elapsed_times = torch.tensor(elapsed_times, device=device)
print(f"Time taken for matrix multiplication: {elapsed_times.mean():.4f} ± {elapsed_times.std():.4f} seconds")

# %%
