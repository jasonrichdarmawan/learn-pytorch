import math
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
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k_size(-1) is hs
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1,2).contiguous().view(B, T, C) 
    # re-assemble all head outputs side by side
    # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs)
    # output projection
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    # pytorch issue #39853 (because the error function erf was slow in tensorflow some years ago, so hendrycks use tanh approximation)
    # GPT-2 use tanh approximation
    # Lllama 3 use SwiGLU
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  pass
  # device = "mps" # maybe has issue
print(f"using device: {device}")

train_loader = DataLoaderLite(B=4, T=32)

# get logits
model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
  x, y = train_loader.next_batch()
  x, y = x.to(device), y.to(device)
  optimizer.zero_grad()
  logits, loss = model(x, y)
  loss.backward()
  optimizer.step()
  print(f"step {i}, loss: {loss.item()}")

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