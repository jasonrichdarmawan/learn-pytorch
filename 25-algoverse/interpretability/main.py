# %%

import sys
from argparse import ArgumentParser
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# %%

print("Setting up reproducibility")
torch.manual_seed(0)

# %%

print("Setting up environment")

if False:
  print("Only run this in Jupyter Notebook")
  print("Simulating environment...")
  WORKSPACE_PATH = "/root/autodl-fs"
  sys.argv = [
    "main.py",
    "--models_path", f"{WORKSPACE_PATH}/transformers",
  ]

parser = ArgumentParser()
parser.add_argument(
    "--models_path",
    type=str,
    help="Path to the directory where models are stored",
)
args = parser.parse_args().__dict__

# %%

print("Loading GPT-2 model")
model_path = os.path.join(
  args["models_path"],
  "gpt2",
)
model = AutoModelForCausalLM.from_pretrained(
  model_path,
)
tokenizer = AutoTokenizer.from_pretrained(
  model_path,
)

# %%

print("Wrapping GPT-2 model with HookedTransformer")
model = HookedTransformer.from_pretrained(
  model_name="gpt2",
  hf_model=model,
  tokenizer=tokenizer,
)

# %%

print("Test a forward pass")
prompt = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text=prompt)
print(f"len(inputs['input_ids]): {len(inputs['input_ids'])}")

# %%

print("Use its caching mechanism to run the prompt and retrieve the hidden states for every layer")
logits, cache = model.run_with_cache(
  input=prompt,
)
print(f"logits.shape: {logits.shape}")
print(f"logits:\n{logits}")
print(f"cache:\n{cache}")

# %%

print("Identify the unembedding weight matrix from the Hugging Face model")
W_U = model.unembed.W_U
print(f"Unembedding matrix W_U shape: {W_U.shape}")

# %%

def func():
  print("Test matrix multiplication with the unembedding matrix")
  hidden_state = cache["blocks.11.hook_resid_post"][0, 1, :]
  print(f"hidden_state shape: {hidden_state.shape}")
  projection = hidden_state @ W_U
  print(f"projection shape: {projection.shape}")

  print("Test softmax function on the projection")
  softmax_projection = F.softmax(projection, dim=-1)
  print(f"softmax_projection shape: {softmax_projection.shape}")

  print("Get the probability assigned to the token ' fox'")
  fox_token_id = tokenizer.encode(" fox")[0]
  fox_probability = softmax_projection[fox_token_id]
  print(f"Probability assigned to ' fox': {fox_probability.item():.2f}")
func()

# %%

fox_token_id = tokenizer.encode(" fox")[0]
print(f"Token ID for ' fox': {fox_token_id}")

fox_probabilities = {}
for layer_index in range(model.cfg.n_layers):
  hidden_state = cache[f"blocks.{layer_index}.hook_resid_post"][0, 1, :]
  projection = hidden_state @ W_U
  probs = F.softmax(projection, dim=-1)
  fox_probability = probs[fox_token_id]
  fox_probabilities[layer_index] = fox_probability.item()

# %%

print("Probabilities assigned to ' fox' by each layer:")
for layer_index, prob in fox_probabilities.items():
  print(f"Layer {layer_index}: {prob}")

max_prob_layer = max(fox_probabilities, key=fox_probabilities.get)
max_prob_value = fox_probabilities[max_prob_layer]
print(f"Max probability assigned to ' fox': Layer {max_prob_layer} with probability {max_prob_value}")

# %%