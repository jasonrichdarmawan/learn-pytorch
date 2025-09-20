# %%

USE_CUSTOM_TRANSFORMERS_LIBRARY = True

# %%

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# %%

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# %% [markdown]
"""
Change the dtype to torch.float32 / torch.float16 
/ torch.bfloat16 to see the difference

To see the difference of a token's hidden layer 
activation value, comment the padding logic

FP32 difference is negligible. Meanwhile,
FP16 and BF16 difference are big.
"""

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = torch.float32
device_map = "auto"
layer = 0

# %%

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    dtype=dtype,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
# experiment 1: process in batch

ex1_start = time.time()

ex1_batch_data = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGeorge Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.<|eot_id|>",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGeorge Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition",
]

ex1_inputs = tokenizer(
    ex1_batch_data,
    return_tensors="pt",
    padding=True,
    padding_side="right",
    add_special_tokens=False,
).to(model.device)

ex1_captured = {}
ex1_hook_handle = model.model.layers[layer].register_forward_hook(
    lambda model, input, output, d=ex1_captured: d.update({"output": output})
)
_ = model(**ex1_inputs)
ex1_hook_handle.remove()

ex1_difference = (ex1_captured["output"][0, 18] - ex1_captured["output"][1, 18]).sum()
print("ex1_difference:", ex1_difference)

ex1_end = time.time()
print(f"ex1 time: {ex1_end - ex1_start:.2f}s")

# %%
# experiment 2: process input separately

ex2_start = time.time()

ex2_data1 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGeorge Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.<|eot_id|>"
ex2_data2 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGeorge Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition"

ex2_inputs1 = tokenizer(
    ex2_data1,
    return_tensors="pt",
    # max_length=300,
    # padding="max_length",
    # padding_side="right",
    add_special_tokens=False,
).to(model.device)

ex2_inputs2 = tokenizer(
    ex2_data2,
    return_tensors="pt",
    # max_length=300,
    # padding="max_length",
    # padding_side="right",
    add_special_tokens=False,
).to(model.device)

ex2_captured1 = {}
if USE_CUSTOM_TRANSFORMERS_LIBRARY:
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_pre_hook = (
        lambda self, q, k, cos, sin, d=ex2_captured1: (
            d.setdefault(f"model.layers.{layer}.self_attn", {}),
            d[f"model.layers.{layer}.self_attn"].update(
                {
                    "q_pre": q.detach().clone(),
                    "k_pre": k.detach().clone(),
                    "cos": cos.detach().clone(),
                    "sin": sin.detach().clone(),
                }
            ),
        )
    )
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_post_hook = (
        lambda self, q, k, d=ex2_captured1: (
            d.setdefault(f"model.layers.{layer}.self_attn", {}),
            d[f"model.layers.{layer}.self_attn"].update(
                {"q_post": q.detach().clone(), "k_post": k.detach().clone()}
            ),
        )
    )


def self_attn_hook(module, input, kwargs, output, d=ex2_captured1):
    d.setdefault(f"model.layers.{layer}.self_attn", {})
    d[f"model.layers.{layer}.self_attn"].update(
        {
            "input": kwargs["hidden_states"].detach().clone(),
            "output": output[0].detach().clone(),
        }
    )


ex2_hook_handle1 = [
    model.model.embed_tokens.register_forward_hook(
        lambda model, input, output, d=ex2_captured1: d.update(
            {"model.embed_tokens": {"output": output.detach().clone()}}
        )
    ),
    model.model.rotary_emb.register_forward_hook(
        lambda model, input, output, d=ex2_captured1: d.update(
            {"model.rotary_emb": {"output": output}}
        )
    ),
    model.model.layers[layer].input_layernorm.register_forward_hook(
        lambda model, input, output, d=ex2_captured1: d.update(
            {
                f"model.layers.{layer}.input_layernorm": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
    model.model.layers[layer].self_attn.register_forward_hook(
        self_attn_hook, with_kwargs=True
    ),
    model.model.layers[layer].post_attention_layernorm.register_forward_hook(
        lambda model, input, output, d=ex2_captured1: d.update(
            {
                f"model.layers.{layer}.post_attention_layernorm": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
    model.model.layers[layer].register_forward_hook(
        lambda model, input, output, d=ex2_captured1: d.update(
            {
                f"model.layers.{layer}": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
]
_ = model(**ex2_inputs1)
if USE_CUSTOM_TRANSFORMERS_LIBRARY:
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_pre_hook = None
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_post_hook = None
for hook in ex2_hook_handle1:
    hook.remove()

ex2_captured2 = {}
if USE_CUSTOM_TRANSFORMERS_LIBRARY:
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_pre_hook = (
        lambda self, q, k, cos, sin, d=ex2_captured2: (
            d.setdefault(f"model.layers.{layer}.self_attn", {}),
            d[f"model.layers.{layer}.self_attn"].update(
                {
                    "q_pre": q.detach().clone(),
                    "k_pre": k.detach().clone(),
                    "cos": cos.detach().clone(),
                    "sin": sin.detach().clone(),
                }
            ),
        )
    )
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_post_hook = (
        lambda self, q, k, d=ex2_captured2: (
            d.setdefault(f"model.layers.{layer}.self_attn", {}),
            d[f"model.layers.{layer}.self_attn"].update(
                {"q_post": q.detach().clone(), "k_post": k.detach().clone()}
            ),
        )
    )


def self_attn_hook(module, input, kwargs, output, d=ex2_captured2):
    d.setdefault(f"model.layers.{layer}.self_attn", {})
    d[f"model.layers.{layer}.self_attn"].update(
        {
            "input": kwargs["hidden_states"].detach().clone(),
            "output": output[0].detach().clone(),
        }
    )


ex2_hook_handle2 = [
    model.model.embed_tokens.register_forward_hook(
        lambda model, input, output, d=ex2_captured2: d.update(
            {"model.embed_tokens": {"output": output.detach().clone()}}
        )
    ),
    model.model.rotary_emb.register_forward_hook(
        lambda model, input, output, d=ex2_captured2: d.update(
            {"model.rotary_emb": {"output": output}}
        )
    ),
    model.model.layers[layer].input_layernorm.register_forward_hook(
        lambda model, input, output, d=ex2_captured2: d.update(
            {
                f"model.layers.{layer}.input_layernorm": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
    model.model.layers[layer].self_attn.register_forward_hook(
        self_attn_hook, with_kwargs=True
    ),
    model.model.layers[layer].post_attention_layernorm.register_forward_hook(
        lambda model, input, output, d=ex2_captured2: d.update(
            {
                f"model.layers.{layer}.post_attention_layernorm": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
    model.model.layers[layer].register_forward_hook(
        lambda model, input, output, d=ex2_captured2: d.update(
            {
                f"model.layers.{layer}": {
                    "input": input[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            }
        )
    ),
]
_ = model(**ex2_inputs2)
if USE_CUSTOM_TRANSFORMERS_LIBRARY:
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_pre_hook = None
    model.model.layers[layer].self_attn.apply_rotary_pos_emb_post_hook = None
for hook in ex2_hook_handle2:
    hook.remove()

ex2_rotary_emb_cos_difference = (
    (
        ex2_captured1["model.rotary_emb"]["output"][0][0, 18]
        - ex2_captured2["model.rotary_emb"]["output"][0][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_rotary_emb_cos_difference:", ex2_rotary_emb_cos_difference)

ex2_rotary_emb_sin_difference = (
    (
        ex2_captured1["model.rotary_emb"]["output"][1][0, 18]
        - ex2_captured2["model.rotary_emb"]["output"][1][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_rotary_emb_sin_difference:", ex2_rotary_emb_sin_difference)

ex2_embed_tokens_difference = (
    (
        ex2_captured1["model.embed_tokens"]["output"][0, 18]
        - ex2_captured2["model.embed_tokens"]["output"][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_embed_tokens_difference:", ex2_embed_tokens_difference)

if USE_CUSTOM_TRANSFORMERS_LIBRARY:
    ex2_layer_self_attn_cos_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["cos"][0, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["cos"][0, 18]
        )
        .abs()
        .sum()
    )
    print("ex2_layer_self_attn_cos_difference:", ex2_layer_self_attn_cos_difference)

    ex2_layer_self_attn_sin_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["sin"][0, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["sin"][0, 18]
        )
        .abs()
        .sum()
    )
    print("ex2_layer_self_attn_sin_difference:", ex2_layer_self_attn_sin_difference)

    ex2_layer_self_attn_q_pre_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["q_pre"][0, :, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["q_pre"][0, :, 18]
        )
        .abs()
        .sum()
    )
    print("ex2_layer_self_attn_q_pre_difference:", ex2_layer_self_attn_q_pre_difference)

    ex2_layer_self_attn_k_pre_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["k_pre"][0, :, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["k_pre"][0, :, 18]
        )
        .abs()
        .sum()
    )
    print("ex2_layer_self_attn_k_pre_difference:", ex2_layer_self_attn_k_pre_difference)

    ex2_layer_self_attn_q_post_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["q_post"][0, :, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["q_post"][0, :, 18]
        )
        .abs()
        .sum()
    )
    print(
        "ex2_layer_self_attn_q_post_difference:", ex2_layer_self_attn_q_post_difference
    )

    ex2_layer_self_attn_k_post_difference = (
        (
            ex2_captured1[f"model.layers.{layer}.self_attn"]["k_post"][0, :, 18]
            - ex2_captured2[f"model.layers.{layer}.self_attn"]["k_post"][0, :, 18]
        )
        .abs()
        .sum()
    )
    print(
        "ex2_layer_self_attn_k_post_difference:", ex2_layer_self_attn_k_post_difference
    )

ex2_layer_input_layernorm_input_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.input_layernorm"]["input"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.input_layernorm"]["input"][0, 18]
    )
    .abs()
    .sum()
)
print(
    "ex2_layer_input_layernorm_input_difference:",
    ex2_layer_input_layernorm_input_difference,
)

ex2_layer_input_layernorm_output_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.input_layernorm"]["output"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.input_layernorm"]["output"][0, 18]
    )
    .abs()
    .sum()
)
print(
    "ex2_layer_input_layernorm_output_difference:",
    ex2_layer_input_layernorm_output_difference,
)

ex2_layer_self_attn_input_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.self_attn"]["input"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.self_attn"]["input"][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_layer_self_attn_input_difference:", ex2_layer_self_attn_input_difference)

ex2_layer_self_attn_output_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.self_attn"]["output"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.self_attn"]["output"][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_layer_self_attn_output_difference:", ex2_layer_self_attn_output_difference)

ex2_layer_post_attention_layernorm_input_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.post_attention_layernorm"]["input"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.post_attention_layernorm"]["input"][
            0, 18
        ]
    )
    .abs()
    .sum()
)
print(
    "ex2_layer_post_attention_layernorm_input_difference:",
    ex2_layer_post_attention_layernorm_input_difference,
)

ex2_layer_post_attention_layernorm_output_difference = (
    (
        ex2_captured1[f"model.layers.{layer}.post_attention_layernorm"]["output"][0, 18]
        - ex2_captured2[f"model.layers.{layer}.post_attention_layernorm"]["output"][
            0, 18
        ]
    )
    .abs()
    .sum()
)
print(
    "ex2_layer_post_attention_layernorm_output_difference:",
    ex2_layer_post_attention_layernorm_output_difference,
)

ex2_layer_input_difference = (
    (
        ex2_captured1[f"model.layers.{layer}"]["input"][0, 18]
        - ex2_captured2[f"model.layers.{layer}"]["input"][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_layer_input_difference:", ex2_layer_input_difference)

ex2_layer_output_difference = (
    (
        ex2_captured1[f"model.layers.{layer}"]["output"][0, 18]
        - ex2_captured2[f"model.layers.{layer}"]["output"][0, 18]
    )
    .abs()
    .sum()
)
print("ex2_layer_output_difference:", ex2_layer_output_difference)

ex2_end = time.time()
print(f"ex2 time: {ex2_end - ex2_start:.2f}s")

# %%
# experiment 3: tokenize target separately

ex3_start = time.time()

ex3_batch_data = [
    {
        "question": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.<|eot_id|>",
    },
    {
        "question": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition",
    },
]

ex3_inputs1 = tokenizer(
    ex3_batch_data[0]["question"] + ex3_batch_data[0]["answer"],
    return_tensors="pt",
    max_length=300,
    padding="max_length",
    padding_side="right",
    add_special_tokens=False,
).to(model.device)

ex3_input_tok2 = tokenizer(
    ex3_batch_data[1]["question"],
    return_tensors="pt",
    add_special_tokens=False,
)
ex3_target_tok2 = tokenizer(
    ex3_batch_data[1]["answer"],
    return_tensors="pt",
    add_special_tokens=False,
)
ex3_input_ids2 = torch.cat(
    [
        ex3_input_tok2["input_ids"],
        torch.unsqueeze(ex3_target_tok2["input_ids"][0], dim=0),
    ],
    dim=1,
).to(model.device)
ex3_attention_mask = torch.cat(
    [
        ex3_input_tok2["attention_mask"],
        torch.unsqueeze(ex3_target_tok2["attention_mask"][0], dim=0),
    ],
    dim=1,
).to(model.device)

pad_len = 300 - ex3_input_ids2.shape[1]
if pad_len > 0:
    ex3_input_ids2 = F.pad(
        ex3_input_ids2,
        pad=(0, pad_len),  # (dim1_left, dim1_right)
        value=tokenizer.pad_token_id,
    )
    ex3_attention_mask = F.pad(
        ex3_attention_mask,
        pad=(0, pad_len),  # (dim1_left, dim1_right)
        value=0,
    )

ex3_inputs2 = {
    "input_ids": ex3_input_ids2,
    "attention_mask": ex3_attention_mask,
}

ex3_captured1 = {}
ex3_hook_handle1 = model.model.layers[layer].register_forward_hook(
    lambda model, input, output: ex3_captured1.update({"output": output})
)
_ = model(**ex3_inputs1)
ex3_hook_handle1.remove()

ex3_captured2 = {}
ex3_hook_handle2 = model.model.layers[layer].register_forward_hook(
    lambda model, input, output: ex3_captured2.update({"output": output})
)
_ = model(**ex3_inputs2)
ex3_hook_handle2.remove()

ex3_difference = (ex3_captured1["output"][0, 18] - ex3_captured2["output"][0, 18]).abs().sum()
print("ex3_difference:", ex3_difference)

ex3_end = time.time()
print(f"ex3 time: {ex3_end - ex3_start:.2f}s")

# %%
