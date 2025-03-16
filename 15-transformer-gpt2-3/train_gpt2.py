import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from hellaswag import render_example, iterate_examples

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
    # version 1:
    # not necessary with F.scaled_dot_product_attention
    # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
  
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
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean")
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
    # the relationships between the weight decay, learning rate, batch size, the atom parameters beta 1, beta 2 and epsilon
    # these are very complicated mathematical relationships in the optimization literature
    # for the most part I'm in this video I'm just trying to copy paste the settings that OpenAI use, this is a complicated topic quite deep
    # note: for different models we of course have different hyper parameters for the transformer that dictate the size of the transformer network
    # we also have different learning rate, so we're seeing the pattern that the biggner network are trained with slightly lower learning rate
    # we also see this batch size, where in the small network they use a smalelr batch size and in the bigger network they use a bigger batch size
    # now the problem for us we can't just use 0.5 million batch size (batch size is referring the number of tokens or roughly B=0.5e6 / T=0.5e6 / 1024=488)
    # the problem is that i can't come in here and set this to 488 because my GPU would explode this woudl not fit for sure
    # and so but we still want to use this batch size because again the batch size is correlated with all of the other hyper parameters and the learning rate and so on
    # so we want to have a faithful representation of all the hyper parameters and therefore we need to use a batch size of 0.5 million tokens
    # the question how do we use 0.5 million if we only have small GPU?
    # for that, we need to use what's called gradient accumulation
    # and it allows us to simulate in a serial way any arbitrary batch size of 0.5 million we just have to run longer
    # and we have to process multiple sequences and basically add up all the gradients fro mthem to simulate a batch size of .5 million
    return optimizer

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import tiktoken
import numpy as np

def load_tokens(filename: str) -> torch.Tensor:
  arr = np.load(filename)
  arr = arr.astype(np.int32)
  # earlier version of PyTorch may have difficulty converting from uint16 to long. Inside `load_tokens`, we added
  # `npt = npt.astype(np.int32)` to use numpy to convert uint16 to int32 before converting to torch tensor and then covnerting to long
  ptt = torch.tensor(arr, dtype=torch.long)
  return ptt

class DataLoaderLite:
  # dataset:
  # RedPajama
  # SlimPajama: 627B token (clean and deduplicated version of RedPajama)
  # FineWeb: 15 trillion tokens
  # FineWeb-Edu: 1.3 trillion (very high educational content) and 5.4 trillion (high educational content)
  #   this dataset is filtered by Llama 3 70bB
  #   we are going to use this sample 10 billion tokens subsample of it because we're not going to be training on trillions of tokens
  # we're just going to train on 10 billion sampel of Fine-Web Edu
  # this is suffices to really get close to GPT2 performance
  # we are going to use sample-10BT of FineWeb-Edu
  # download FineWeb Edu from huggignface
  # preprocess and pretokenize all of the data
  # it will save data shards to a folder on local disk
  def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}

    # get the shard filenames
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f"no shards found split {split}"
    if master_process:
      print(f"found {len(shards)} shards for split {split}")
    self.reset()

    # tiny shakespear
    # # at init load tokens from disk and store them in memory
    # with open("input.txt", "r") as f:
    #   text = f.read()
    # enc = tiktoken.get_encoding("gpt2")
    # tokens = enc.encode(text)
    # self.tokens = torch.tensor(tokens)
    # print(f"loaded {len(self.tokens)} tokens")
    # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    # state (tiny shakespear)
    # self.current_position = self.B * self.T * self.process_rank
    # what we want is we want to stride out all the processes
    # so one way to do this is we basically take self.B * self.T * self.process_rank
    # process 0 will start at 0, process 1 will start at B * T, process 2 will start at 2 * B * T, etc.
  
  def reset(self):
    # state, init at shard zero
    # Fine-Web-Edu: 10 billion tokens, 100 shards
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = self.B * self.T * self.process_rank
    # what we want is we want to stride out all the processes
    # so one way to do this is we basically take self.B * self.T * self.process_rank
    # process 0 will start at 0, process 1 will start at B * T, process 2 will start at 2 * B * T, etc.

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1].view(B, T)) # inputs
    y = (buf[1:]).view(B, T) # targets
    # advance the position in the tensor
    self.current_position += B * T * self.num_processes
    # if loading the next batch would be out of bounds, reset
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
      # FineWeb-Edu: 10 billion tokens, 100 shards
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.current_shard])

      self.current_position = self.B * self.T * self.process_rank
    # ddp lesson
    # check the loss with single GPU and the loss with DDP
    # the numbers will not exactly match up.
    # the reason in the data loader, we are just iterating through batches in slightly diferent way
    # because now we're looking for an entire page of data and if that page for all the gpu if that chunk exceeds the number of tokens
    # we just loop
    # and so actually the singgle GPU and the 8 GPU process will end up resetting in a slighlty different manner
    # and so our batches are slightly different
    # but one way to convince yourself that this is okay
    # just make the total_batch_size much smaller and the B and T
    # and then so i think i use 4*1024*8 = 32,768 as total_batch_size, B=4, T=1024 and then I made sure that the single GPU will do 8 gradient accumulation steps
    # and then you reduce the boundary effects of the data loader and you'll see the data match up.
    return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor):
  # evaluate the autoregressive loss at all positions
  shift_logits = (logits[..., :-1, :]).contiguous()
  shift_tokens = (tokens[..., 1:]).contiguous()
  flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
  flat_shift_tokens = shift_tokens.view(-1)
  shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
  shift_losses = shift_losses.view(tokens.size(0), -1)
  # now get the average loss just for the completion region (where mask == 1), in each row
  shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
  masked_shift_losses = shift_losses * shift_mask
  # sum and divide by the number of 1s in the mask
  sum_loss = masked_shift_losses.sum(dim=1)
  avg_loss = sum_loss / shift_mask.sum(dim=1)
  # now we have a loss for each of the 4 completions
  # the one with the lowest loss should be the most likely
  pred_norm = avg_loss.argmin().item()
  return pred_norm

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAl_RANK, and WORLD_SIZE
# so the tricky thing with running multiple processes is you always have to iamgine that there is going to be 8 processes running in parallel
# so as you read the code now you have to imagine there is 8 python interpreters running down these line of code
# and the only difference between them is that they have different DDP rank 
# so they all come here they all pick the exact same seed they all make these calculations completely unaware of the other copies running roughly speaking
# so they make the exact same calcualtion and now we have to adjust these calculations to take into account that there is certain world size and certain ranks
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
  # use of DDP atm demands CUDA, we set the device appropriately according to rank
  assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
  init_process_group(backend="nccl")
  ddp_rank = int(os.environ["RANK"])
  ddp_local_rank = int(os.environ["LOCAL_RANK"])
  # local rank is something that is only used in multi-node setting
  # we only have 1 node with 8 GPUs, so local rank is the rank of the GPU on a single node, so from 0 to 7 as an example
  # but for us we are mostly are going to be running on a single box so the things we care about are rank and world size
  # world_size is 8
  # rank will be whatever is depending on the GPU that this particular script runs on.
  ddp_world_size = int(os.environ["WORLD_SIZE"])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
  # master process is arbitrarly process number 0
  # the other process are thought of mostly as a compute processes that are assiting
  # so Master process will have some other additional work to do
  # all the other processes will mostly just be doing forward backward
else:
  # vanilla, non-DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  # attempt to autodetect the device
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    # maybe has issue, i was trying to overfit the model on a single batch (the data is not changed) for 50 steps
    # but it wasn't overfitting. it overfit when I use CPU though.
  print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size= 524288 # 2**19, ~0.5M, in number of tokens
# see "Language Models are Few-Shot Learners" paper
# GPT-3 Small batch size is 0.5M tokens
B = 64 # micro batch size
# lesson 1:
# 64*1024*8 = 524,288 tokens
# if this fits, so that means we would not even be doing gradient accumulation if this ffits
# because this just multiplies out the full total_batch_size
# so no gradient accumulation, and that would run pretty quickly if that fits
# i mean if this works, this is basically a serious pre-training run
# we are not logging, we are not evaluating the validation split, we are not running any evaluations yet
# but if we let this run for a while we are going to actually get a pretty good model and the model might be on par with or better than gpt-2 124M
# everything here looks good
# we're doing 330 milliseconds per iteration
# and we have to do a total of 19073 iterations * 0.33/60/60 = 1.47 hours
# so 1.5 hours run like this and we don't have to do use gradient accumulation which is nice
# you might not have that luxury in your GPU in that case just start decreasing the batch size until things fit
# but keep it to nice numbers
# lesson 2:
# now because we have the total batch size and the graident accumulation steps
# our settings of B is purely a performance optimization kind of settings
# so if you have a big GPu you can actually increase this to 32 and you'll probably
# go a bit faster if you have a very small GPU you can try eight or four
# but in any case you should be getting the exact same optimization and the same answers
# up to like a floating point error because the gradient accumulation kicks in and can
# handle everything serially as necessary
T = 1024 # sequence length
# lesson 2:
# if you wanted to exactly be faithful to GPT-3 you would also want to make the following 
# the sequence length of GPT-3 is 2x, set T to 2048
# and if you want the exact same number of tokens, 0.5M per iterations or per step
# you want to decrease B to 32. so they still multiply to 0.5M
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# 16*1024*8 = 131,072 tokens on a single forward backward on the 8 GPUs
if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# print("I am GPU ", ddp_rank)
# print("Bye")
# import sys; sys.exit(0)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

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

# create model
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
use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
# https://github.com/karpathy/build-nanogpt/issues/79
# IggShaman said when a model is compiled, it will fix its input and output tensor sizes.
if use_compile:
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
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
  # in a forward pass it actually behaves identically
  # my understanding of it is nothing should be changed in the forward pass
  # but in the backward pass , as you are doing the backward pass in the simplest setting
  # once the backward pass is over, on each independent GPU
  # each independent GPU has the gradient for al lthe parameters
  # and what DDP does for you is once the backward pass is over, it will call, what's called all reduce and it basically does an average
  # across all the ranks of their gradients and then it will deposit that average on every single rank
  # so every single rank will end up with the average on it
  # so basically that's the communciation it just synchornizes and averages the gradients and that's what DDP offers you
  # now DDP actually is a little bit more involved than that because as you doing the backward pass through the layers of the Transformer
  # it actually can dispatch communications for the gradients while the backward pass is still happening
  # so there is an overlap of the communication of the gradients and the syncchronization of them and the backward pass
  # it's just more efficient and to do it that way
  # so that's what DDP does for you
  # forward is unchanged
  # backward is mostly unchanged
  # and we're tackling on this average as we'll see in a bit
  # prev: 93ms, 176k tokens per second
  # now: 356ms, 1.47m tokens per second
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
# i saw some people already play with this a little bit
# in a previous related repository
# it turns out that you can actually almost like 3x these
# so it's possible that the maximum learning rate can alot higher
# and for some reason the GPT 3 hyper parameter that are we inherting are actually exteremly conservation
# and you can actually can get away with higher learning rate and it would train faster
# so a lot of these hyperparameters are quite tunable and feel free to play with them
# and they probably are not set precisely correctly and it's possible that you can get away with doing this basically
min_lr = max_lr * 0.1
warmup_steps = 715
# GPT-3 paper says that they warmup the learning over 375 million tokens
# 375e6 / 2**19 = 715 steps
max_steps = 19073 # 19,073 steps is ~1 epoch of sample 10B of FineWeb-Edu, if data is 10B tokens and batch size 0.5M tokens
# 2**19 = 524,288 tokens per step
# we to do 10e9 / 10 billion tokens
# 10e9 / 2**19 = 19,073 steps
# note: max_steps = 19073 * 4
# means 40B tokens, 8 hours, the hellaswag eval is similar to GPT-3
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

# this iscalled, the main training loop
# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, beta=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log directory we will write checkpoints to and log to
# which will record the train loss, validation loss, and hellaswag accuracy
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
  pass

for step in range(max_steps):
  t0 = time.time()
  last_step = (step == max_steps - 1)

  # once in a while evaluate oru validation loss
  # every 100 iterations including the zeroth iteration we put the model into evaluation mode
  # we reset the val loader
  # no gradients involved
  # we are going to basically accumulate over say 20 steps and then average it all up and print out the validation loss
  # and so that basically is the exact same logic as the training loop
  # but there is no loss that backward it's only inference, we are just measuring the loss
  # we're adding it up everything else otherwise applies and is exactly as we've seen it before
  # so that's nice that would tell us some amount some a little bit how much we're overfitting
  # that said like we have roughly infinity data
  # so we're mostly expecting our train and val loss to be about the same
  # but the other reason i'm kind of interested in this because we can take the GPT-2 124M as OpenAI released it
  # we can initialize from it and we can basically see what kind of loss it achieves on the validaiton loss as well and it gives us 
  # kind of an indication as to how much that model would generalize to FineWeb-Edu validaiton splits
  # that said it's not super fair comparison to GPT-2 because it was trained on a very different data distribution
  # but it's still kind of like an interesting data point
  # and in any case you would always want have a validation split in a training run like this so that you can make sure that you are not
  # overfitting and this is especially a concern if we were to make more Epoch in our training data 
  # right now, we are just doing a single epoch, but if we get to a point where we want to train on 10 epochs or something like that
  # we would be really careful with maybe we are memorizing that data too much if we have big enough model
  # and our validation splits would be one way to tell whether that is happening
  if step % 250 == 0 or last_step:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f"validation loss: {val_loss_accum.item():.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
      if step > 0 and (step % 5000 == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
          'model': raw_model.state_dict(),
          'config': raw_model.config,
          'step': step,
          'val_loss': val_loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and rng seeds etc.,
        # if you wanted to more exactly resume training
        # because the optimizer have few additional buffers because of AdamW
        # it's got the M and V, you need to resume the optimizer properly
        # you have to be careful with the rng seeds, random number generator and so on
        # so if you wanted to be exactly to resume optimization you have to think through the state of the training process
        # but if you just want to save the model this is how you would do it
        # and one nice reason why you might want to do this is because you might want to evaluate a lot mroe carefully
        # so here we are only kind of like winging the hellaswag eval, but you may want to use something nicer
        # for example:
        # Luther evaluation harness, so this is a way to also evaluate languiage models
        # so it's possible that you may want to use basically different infrastructure to more thoroughly evaluate the model
        # on different evaluations and compare it to the openai gpt2 model on many other tasks like for example that involve
        # math, code or different languages and so on. so that is a nice functionality to have as well
        # and then the other thing i want to mention is that, everything we build here is just the pre-training steps.
        # so, the GPT here is a it dreams documents, it just predicts the next token, you can't talk to it
        # like you can talk to chatgpt
        # if you want to talk to the model, we have to finetune it to the chat format and it's not actually like that complicated
        # if you're looking at supervised fine tuning or SFT. really what that means is we're just we're just swapping out
        # the dataset into a dataset that is alot more conversational there is a "user", "assistant", "user", "assistant" kind of structure
        # and we just fine tune on it. and we basically fill in the user tokens and sample the assistant tokens
        # it's not alot more deeper than that, basically we swap out the dataset and continue training
        # but for now we're going to stop at pre-training
        torch.save(checkpoint, checkpoint_path)
  
  # evaluate hellaswag
  if (step % 250 == 0 or last_step) and (not use_compile):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
      # only process examples where i % ddp_world_size == ddp_rank
      if i % ddp_world_size != ddp_rank:
        continue
      # render the example into tokens and labels
      _, tokens, mask, label = render_example(example)
      tokens = tokens.to(device)
      mask = mask.to(device)

      pad_x_to = (T if use_compile else tokens.size(1)) - tokens.size(1)
      tokens = torch.cat((tokens, torch.zeros([tokens.size(0), pad_x_to], dtype=torch.long)), dim=1)
      mask = torch.cat((mask, torch.zeros([mask.size(0), pad_x_to], dtype=torch.long)), dim=1)

      # get the logits
      with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        # get the most likely option with the lowest loss
      num_total += 1
      num_correct_norm += int(pred_norm == label)
    
    # reduce the stats across all processes
    if ddp:
      num_total = torch.tensor(num_total, dtype=torch.long, device=device)
      num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
      dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
      num_total = num_total.item()
      num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
      print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} hella {acc_norm:.4f}\n")
  
  # once in a while generate from the model (except step 0, which is noise)
  # disabled because torch.compile throws a scary error i can't solve rn
  # if you dsiale torch.compile, this code works fine
  if (step > 0 and step % 250 == 0) or last_step:
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    at_idx = tokens.size(1)
    pad_x_to = (T if use_compile else max_length) - tokens.size(1)
    tokens = torch.cat((tokens, torch.zeros([tokens.size(0), pad_x_to], dtype=torch.long)), dim=1)

    # pad the y axis
    if B != num_return_sequences and use_compile:
      # When B is smaller than the number of examples we generate, we can simply
      # loop over the example generation code.
      assert num_return_sequences <= B, f"TODO: {num_return_sequences=} is > {B=}; add support for that"
      tokens = F.pad(tokens, (0, 0, 0, B - num_return_sequences), 'constant', enc.eot_token)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    # the generator object in pytorch
    # so that i have direct control over the sampling of the random number
    # because i don't want to impact the rng state of the random number generator
    # that is the global one used for training
    # i want this to be completely outside the training loop
    # and so i am using a special sampling rng
    # and then i make sure to seed it that every rank have different seed
    # you will notice that we are running a bit slower
    # that's because i actually had to disable torch.compile to get this to sample
    # so we're running a bit slower
    # so for some reason it works with no torch.compile but when I torch.compile my model
    # i get a really scary error from pytorch and i have no idea how to resolve it right now
    sample_rng.manual_seed(42 + ddp_rank)
    while at_idx < max_length:
      # forward the model to get the logits
      with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, at_idx-1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B,1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        # append to the sequence
        xgen[:, at_idx:at_idx + 1] = xcol # (5, 9)

      at_idx += 1

    # print the generated text
    for i in range(num_return_sequences):
      tokens = xgen[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(f"rank {ddp_rank} sample {i}: {decoded}")

  # training loop
  # do one step of the optimization
  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
      # we want to synchronize the gradients only at the last step
      # confusingly, `model.require_backward_grad_sync` is actually used by both the forward and backward pass.
      # moved up the line so that it also gets applied to the forward pass.
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
    # we have to scale the loss to account for gradient accumulation,
    # because the gradients just add on each successive backward(),
    # addition of gradients corresponds to a SUM in the objective, but
    # instead of a SUM we want mEAN, Scale the loss here so it comes out right
    loss = loss / grad_accum_steps
    # since we are doing gradient accumulation, and using cross entropy with reduction="mean",
    # the "normalizer" is missing, so we have to divide the loss by the number of accumulation steps
    loss_accum += loss.detach()
    # detaching the tensor from the graph
    loss.backward()
    # DDP
    # we just want them adding up and we don't want them to synchornize every single time
    # that would be extremely wasteful, so basically, we want to add them up and then on the very last step
    # when micro steps become grad_accum_steps - 1
    # only at that last step do we want to actually do the allreduce
    # to average up the gradients
    # so to do that the official sanctioned way
    # it's super ugly.
    # ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
    # with ddp.no.sync():
    #   for input in inputs:
    #     ddp(input).backward9)     # no synchronization, accumulate grads
    # ddp(another_input).backward() # synchronize grads
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # the problem: loss_accum is outside the ddp container
    # so that's not being averaged
    # so when we are printing the loss_accum in the master process, rank 0, it is just going to be printing the losses that it saw on its process
    # but instead we want it to print the loss over all the processes and the average of that loss
    # because we did average of gradients so we want the average of loss as well
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
  dt = t1 - t0 # time difference in seconds 
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec = tokens_processed / dt
  if master_process:
    print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
      f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
  destroy_process_group()

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

# tiny shakespear
# import sys; sys.exit(0)

# num_return_sequences = 5
# max_length = 30

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(dim=0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# # generate! right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# while x.size(1) < max_length:
#   # forward the model to get the logits
#   with torch.no_grad():
#     logits = model(x) # (B, T, vocab_size)
#     # take the logits at the last position
#     logits = logits[:, -1, :] # (B, vocab_size)
#     # get the probabilities
#     probs = F.softmax(logits, dim=-1)
#     # do top-k sampling of 50 (huggingface pipeline default)
#     # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#     topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#     # select a token from the top-k probabilities
#     ix = torch.multinomial(topk_probs, 1) # (B,1)
#     # gather the corresponding indices
#     xcol = torch.gather(topk_indices, -1, ix) # (B,1)
#     # append to the sequence
#     x = torch.cat((x, xcol), dim=1) # (5, 9) 

# # print the generated text
# for i in range(num_return_sequences):
#   tokens = x[i, :max_length].tolist()
#   decoded = enc.decode(tokens)
#   print(">", decoded)