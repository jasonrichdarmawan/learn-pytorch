# %%

from dotenv import load_dotenv

if load_dotenv(dotenv_path=".env") == False:
    raise RuntimeError("Failed to load .env file")

# %%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%

import torch
from torch import Tensor

from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

# %%

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# %%

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

print("Vocab size: ", tokenizer.vocab_size, "\n")
print("Embedding size: ", model.config.hidden_size, "\n")
print("pad_token_id: ", tokenizer.pad_token_id)

# %%

messages = [
    [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"}
    ],
    [{"role": "user", "content": "What is your name?"}],
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
).to(model.device)

print("inputs keys:", list(inputs.keys()))
print("input_ids shape: ", inputs["input_ids"].shape)
print("input_ids: \n", inputs["input_ids"], "\n")

# %%

outputs = model(**inputs)
print("Output logits shape: ", outputs.logits.shape)

# %%

outputs = model.generate(**inputs, max_new_tokens=40)

print("Generated sequences shape: ", outputs.shape)

for i in range(len(outputs)):
    print("Raw input ids: \n", inputs["input_ids"][i], "\n")
    print("Attention mask: \n", inputs["attention_mask"][i], "\n")
    print("Decoded input: \n", tokenizer.decode(inputs["input_ids"][i]), "\n")
    print("Decoded output: \n", f"{tokenizer.decode(outputs[i])}", "\n")
    print(
        "Decoded generation: \n",
        f"{tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:])}",
        "\n",
    )

# %%

inputs = tokenizer("hello world", return_tensors="pt")
print("Raw input ids: \n", inputs["input_ids"], "\n")
print("Attention mask: \n", inputs["attention_mask"], "\n")
print("Decoded input: \n", tokenizer.decode(inputs["input_ids"][0]), "\n")

# %%
# remove this after debugging

data = {
    "question": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "answer": "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.<|eot_id|>",
}

target_tok = tokenizer(data["answer"], return_tensors="pt", add_special_tokens=False).to(model.device)
target_ids = target_tok["input_ids"][0]
target_attention_mask = target_tok["attention_mask"][0]

if target_ids[0] == tokenizer.bos_token_id or target_ids[0] == tokenizer.unk_token_id:
    target_ids = target_ids[1:]

input_tok = tokenizer(
    data["question"],
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)

cur_input_ids = input_tok["input_ids"]
cur_attention_mask = input_tok["attention_mask"]
start = 0
end = 40
cur_target_ids = target_ids[start:end]
cur_target_attention_mask = target_attention_mask[start:end]
input_ids = torch.cat(
    [
        cur_input_ids, 
        torch.unsqueeze(cur_target_ids[:-1], dim=0)
    ],
    dim=1
)
attention_mask = torch.cat(
    [
        cur_attention_mask,
        torch.unsqueeze(cur_target_attention_mask[:-1], dim=0),
    ],
    dim=1
)
print("input_ids shape:", input_ids.shape)
print("input_ids:", input_ids)

hparams = {
    "layer_module_tmp": "model.layers.{}",
}

layer = 8
target_init = None
lookup_idsx = [18]

def edit_output_fn(
    cur_out: Float[Tensor, "batch seq h_dim"],
    cur_layer: str,
):
    global target_init
    
    if cur_layer == hparams["layer_module_tmp"].format(layer):
        if target_init is None:
            target_init = cur_out[0, lookup_idsx].detach().clone()

with nethook.TraceDict(
    module=model,
    layers=[
        hparams["layer_module_tmp"].format(layer),
    ],
    retain_output=True,
    edit_output=edit_output_fn,
) as tr:
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

position1 = None
def rotary_hook(model, input, output):
    global position1
    position1 = output

activation1 = None
def layer_hook(model, input, output):
    global activation1
    activation1 = output

hook_handle = [
    model.model.rotary_emb.register_forward_hook(rotary_hook),
    model.model.layers[layer].register_forward_hook(layer_hook),
]

_ = model(input_ids)

for hook in hook_handle:
    hook.remove()

# %%

input_ids2 = tokenizer(
    data["question"] + data["answer"],
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)
print("input_ids2 shape:", input_ids2["input_ids"].shape)
print("input_ids2:", input_ids2["input_ids"])

with torch.no_grad():
    with nethook.TraceDict(
        module=model,
        layers=[
            hparams["layer_module_tmp"].format(layer),
        ],
        retain_output=True,
    ) as tr2:
        _ = model(
            input_ids=(
                input_ids2["input_ids"]
                # [:,:len(input_ids[0])]
            ),
            attention_mask=(
                input_ids2["attention_mask"]
                # [:,:len(input_ids[0])]
            )
        )

position2 = None
def rotary_hook(model, input, output):
    global position2
    position2 = output

activation2 = None
def layer_hook(model, input, output):
    global activation2
    activation2 = output

hook_handle = [
    model.model.rotary_emb.register_forward_hook(rotary_hook),
    model.model.layers[layer].register_forward_hook(layer_hook),
]
_ = model(input_ids2["input_ids"])
for hook in hook_handle:
    hook.remove()

# %%

same = torch.all(input_ids == input_ids2["input_ids"][:, :len(input_ids[0])])
print("same tokens:", same)

position_difference = (
    (position1[0] - position2[0][:, :len(input_ids[0])]).sum(),
    (position1[1] - position2[1][:, :len(input_ids[0])]).sum(),
)
print("embedding difference:", position_difference)

activations_difference = (
    target_init 
    - tr2[hparams["layer_module_tmp"].format(layer)].output[0, lookup_idsx]
).sum(dim=1)
print("activations difference:", activations_difference)

activations_difference2 = (
    activation1[0, lookup_idsx] 
    - activation2[0, lookup_idsx]
)

print("activations difference2:", activations_difference2.sum(dim=1))

# %%