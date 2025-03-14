import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----

@dataclass
class GPTConfig:
  block_size: int = 1024 # max sequence length
  vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
  n_layer: int = 12 # number of layers
  n_head: int = 12 # number of heads
  n_embd: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to the batch
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    qkv: torch.Tensor = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    # attention (materializes the large (T,T) matrix for all the queries and keys)

    # version 1:
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k_size(-1) is hs
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    # version 2: FlashAttention
    # previous: 130ms, 126k tokens per second
    # now: 96ms, 169k tokens per second
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1,2).contiguous().view(B, T, C) 
    # re-assemble all head outputs side by side
    # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs)
    # output projection
    y = self.c_proj(y)
    return y

class TanhGELU(nn.Module):
  """
  equivalent to nn.GELU but way slower

  GPU
  ^
  |
  v
  HBM (equivalent of RAM but on GPU)

  e.g. torch.pow(input, 3)
  the input, has to travel to the GPU, to the cores, and to all the caches and register on the actual chips
  and it has to calculate all the elements to the third and then saves the result back to the HBM
  and this travel time actually causes a lot of issues

  and then we do 0.044715 * torch.pow(input, 3) (round trip again)

  torch.compile is doing "kernel fusion"
  in other words, the equation below will be done in one round trip instead of multiple round trip between GPU <-> HBM

  but there are operation that torch.compile will not find
  For example, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" paper
  FlashAttention is a kernel fusion operation

  the reason torch.compile cannot find is that it requires an algorithmic rewrite of how attention is actually implemented

  FLashAttention is very careful with ohw it orchestrates the computation such hat we have fewer reads and writes
  to the high bandwidth memory and so even though we're doing more flops
  the expensive part is the load and store into HBM and that's what they avoid
  and so in particular they do not ever materialize this N by N attention matrix to HBM.

  the way this is achieved is basically the algorithmic rewrite here elis on this online softmax trick
  and the online softmax trick ocming from a previous paper, shows how you can incrementally evaluate a softmax
  without having to sort of realize all of the inputs to the softmax to do the normalization
  and you do that by having these intermediate variables M and L and there's an update to them that allowws you to evaluate the softmax in an onlnie manner

  memory hierarchy
  GPU SRAM                  SRAM 19 TB/s (20 MB)
  GPU HBM                   HBM: 1.5 TB/s (40 GB)
  Main Memory (CPU DRAM)    DRAM: 12.8 GB/s (>1 TB)
  """
  def forward(self, input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))

class MLP(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    # pytorch issue #39853 (because the error function erf was slow in tensorflow some years ago, so hendrycks use tanh approximation)
    # GPT-2 use tanh approximation
    # Lllama 3 use SwiGLU
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANO_GPT_SCALE_INIT = 1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):
  """
  we do not "gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billions tokens of training, depending of the model size."

  in the early stages of the optimization, again the model is in a very atypical setting and mostly what you're learning is that you're mostly learning to ignore
  the tokens that don't come up in your training set very often, you are learning very simple biases and that kind of a thing and so every single example that put 
  through your network is basically just telling you use these tokens and don't use these tokens
  and so the gradients from every single example are extermely highly correlated they look roughly the same in original parts of the optimization
  because they're all just telling you that these tokens don't appear and these tokens do appear
  and so because the gradients are very similar and they're highly correlated
  **then why are you doing batch size of like millions when if you do a batch size of 32k**
  you're basically getting the exact same gradients early on the training and then later in the optimization once you've learned all the simple stuff
  that's where the actual work starts and that's where the gradients become more decorrelated for examples and that's where they actually offer you sort of statistical power in some sense
  """
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config: GPTConfig = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std += (2 * self.config.n_layer) ** -0.5
        # in a block, there are 2 residual connections
        # so, the scale of the initialization should be 1/sqrt(2 * n_layer)
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      # xaiver init
      # std=1/sqrt(d_model)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx: torch.Tensor, targets: torch.Tensor=None) -> torch.Tensor:
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # forward the token and position embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
    # since we are using GPT-2
    # the position encoding is using nn.Embedding instead of pre-computed sin/cos positional encodings
    tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
    x = tok_emb + pos_emb
    # forward the blocks of the tnrasformer
    for block in self.transformer.h:
      x = block(x)
    # forward the final layer norm and the classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)  
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      # logits: (B*T, vocab_size)
      # targets: (B * T)
    return logits, loss

  @classmethod
  def from_pretrained(cls, model_type: str):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = { 
      'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},         # 124M params
      'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024}, # 350M params
      'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},  # 774M params
      'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600}      # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch intiialized minGPT model
    config = GPTConfig(**config_args)
    model = cls(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a  buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model
  
  def configure_optimizers(self, weight_decay: float, learning_rate: float, device: str) -> torch.optim.AdamW:
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    # prev: 93ms, 176k tokens per second
    # now: 90ms, 182k tokens per second
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import tiktoken

class DataLoaderLite:
  def __init__(self, B: int, T: int):
    self.B = B
    self.T = T
    
    # at init load tokens from disk and store them in memory
    with open("input.txt", "r") as f:
      text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f"loaded {len(self.tokens)} tokens")
    print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    # state
    self.current_position = 0
  
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1].view(B, T)) # inputs
    y = (buf[1:]).view(B, T) # targets
    # advance the position in the tensor
    self.current_position += B * T
    # if loading the next batch would be out of bounds, reset
    if self.current_position + (B * T + 1) > len(self.tokens):
      self.current_position = 0
    return x, y

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# attempt to autodetect the device
import time

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  pass
  # device = "mps" # maybe has issue
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high') 
# A100 (Ampere Series) will use Tensor Float 32 (TF32) instead of FP32 for the input operands.
# Note. the accumulator is FP32.
# FP32: 19.5 TFLOPS
# TF32: 156 TFLOPS
# so we are going from 1,000 milliseconds to 300 milliseconds
# and our throughput are going from 16,000 tokens per seconds to 50,000 tokens per seconds
# we're supposed to be getting 8X roughly
# so let's see what happens
# and that 8X came from here, we are going from 19.5 TFLOPS to 156 TFLOPS
# but we are seeing that our throughput roughly 3x not 8x
# so what happened?
# basically, a lot of this workloads are memory bound
# and so even though the TF32 offers in principle a lot faster throughput
# all of these numbers everywhere are still float32 and its float 32 numbers
# that are being shipped all over the place through the memory system
# and its just costing us way too much time to shuttle around all this
# data. and so even though we've made the multiply itself much faster
# we're memory bound and we are not seeing the full benefit uh that would come from this napkin math here
# that said we are getting a 3X faster throughput and this is free
# single line of code in PyTorch
# all your variable are stil float32 everywhere, it just faster, and it's slightly mroe approxiamte
# but we're not going to notice it basically

# get logits
model = GPT(GPTConfig(vocab_size=50304))
# stupid optimization trick but because vocab_size originally was 50257
# 50257 is an odd number, 50304 is even divisible by 128
# prev: 96ms, 169k tokens per second
# now: 93ms, 176k tokens per second
# it doesn't matter because the model will just learn to drive these fake tokens to zero in probabilities
# cuda so many kernels use block tiles and these block tiles usually nice number
# power of two.
# calculations are done in lie chunks of 64, or chunks 32
# and when your desired calculations doesn't neatly fit int those block tiles
# there are all kinds of bondary kernels that can kick in to do like the last part.
# so basically in alot of kernel, they will chunk up your input and they will do the nice part first
# and they will do the second phase where they come back to any that like remains and then they process the remaining part
# and that kernels for that can be very inefficient
# so you're basically spinning up all this extra compute and this extremely inefficient and then make it fit nicely and usually empirircally 
# that ends up actually running faster.
model.to(device)
model = torch.compile(model)
# this will cost you compilation time
# but as you might guess
# it's going to make the code alot faster
# "speedup mainly comes from reducing python overhead and GPU read/writes"
# so we went from 300 milliseocnds we're now running at 129 milliseconds
# 55k tokens per second to 126k tokens per second 
# torch.compile makes pytorch dobn't need to run in eager mode
# python interpreter normally does it layer by layer in the forward pass
# torch.compile will take out the python interpeter out

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it: int) -> float:
  # note: GPT-3 learning rate is 10% of its max_lr after 260 billion tokens
  # cosine decay learning
  # 1) linear warmup for warmup_iters steps
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  # 2) if it > lr_decay_iters, return min learning rate
  # this is where our implementation is different compared to GPT-3
  if it > max_steps:
    return min_lr
  # 3) in between, use cosien decay down to min learning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)

# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, beta=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
  t0 = time.time()
  x, y = train_loader.next_batch()
  x, y = x.to(device), y.to(device)
  optimizer.zero_grad()
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    # pytorch automatic mixed precision
    # some things pytorch is keeping in float32
    # https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32
    # some things pytorch is converting to lower precision (bfloat16)
    # https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
    # we used to be 333 millisecond, we are now 300
    # we used to be somewhere aroudn 50k tokens per second and now we are at 55k tokens per second 
    # we are paying in precision for this
    # we expect slightly less accurate result
    # with respect to the original fp32
    # but empircally in many cases this is a worth it kind of tradeoff
    # because it allows you to run faster, and you could for exmapel train longer
    # and make up for the lost precision that's bfloat16 for now.
    # try to type logits.dtype
    # import code; code.interact(local=locals())
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # every single gradients on all the parameters, you square it and you add it all up and you take a big square root of that
  # that's the norm of parameter vector basically.
  # it's the length of it, if you like to look it up that way.
  # and we make sure it's length is no more than 1.0 and we are going to clip
  # the reason people like to use it, sometimes you can get unlucky during the optimization, maybe it's a bad data batch or something like that
  # if you get very unlucky in a batch, you might get really high loss, and a really high loss can lead to a really high gradient
  # and this could basically shock your model and shock the optimizer
  # so people like to use gradient norm clipping to prevent the model from basically getting too big of shock in terms of gradients magnitude
  # it's a bit of a hacky solution, it's like a patch on top of like deeper issues but people still do it fairly frequently
  # i like to always visualize because it is useful inforamtion
  # and sometimes you can look at the norm of the gradient 
  # and if it's well behaved things are good
  # if it's climbing things are bad adn they are destablizing during training
  # sometime you could get a spike in the norm and that means there is some kind of an issue or an instability
  
  # determine and set the learning rate for this iteration
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  if device == "cuda":
    torch.cuda.synchronize()
  # when your cpu runs it scheduling work on GPU. So, it sends a request and then it continues running.
  # And so, wit can happen soemtiems that we sort of speed through this and we queue up a lot of kernels to run on the GPU
  # and then the CPU sort of like gets here (t1) and takes time at time but actually the GPU is still running
  # because it takes a time to actually work through the work that was scheduled to run
  # and so you are just building up a queue for the GPU.
  # and so actually if you need to, you want to wait torch.cuda.synchronize() and this will wait for the GPU to finish all the work that was schedueld to run
  # and then we can actually take the time.
  t1 = time.time()
  dt = (t1 - t0) * 1000 # time difference in milliseconds 
  tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
  print(f"step {step} | loss: {loss.item()} | lr: {lr:.4e} | norm: {norm:.4f } | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# lesson 1:
# try to get random x and y and do forward pass, then print the loss
# cross entropy loss is -ln(value)
# so, for sanity check, initially the loss should be -ln(1/50257)
# logits, loss = model(x, y)
# print(loss)

# lesson 2:
# gpt-2 has 50257 tokens, this includes non-English characters
# the loss will come down but not too much
# what's happening is that because in the 50257 tokens many of those tokens never occur
# in our dataset. So, there some very easy gains to be made here in the optimization
# by for example taking the biases of all the logits that never occur and driving them to negative infinity.
# and that would basically just it's that all of crazy unicode or different languages, those tokens 
# never occur so their probability should be very low
# so the gains that we should be seeing are along the lines of basically deleting the usage of tokens that
# never occur that's probably most of the loss gain that we're going to see at this scale right now
# but we shouldn't come to a zero because um
# we are only doing 50 itertions and i don't think that enough to do an epoch right now.
import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(dim=0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
while x.size(1) < max_length:
  # forward the model to get the logits
  with torch.no_grad():
    logits = model(x) # (B, T, vocab_size)
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1)
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
    # select a token from the top-k probabilities
    ix = torch.multinomial(topk_probs, 1) # (B,1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B,1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1) # (5, 9) 

# print the generated text
for i in range(num_return_sequences):
  tokens = x[i, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(">", decoded)