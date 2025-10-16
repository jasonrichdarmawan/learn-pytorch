# %%

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from peft import TaskType, get_peft_model, PeftModel
from peft.tuners.nullspacelora import NullSpaceLoraConfig

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# %%

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "Qwen/Qwen2.5-7B-Instruct"

peft_model_id = "jasonrichdarmawan/llama3-8b-instruct-lora-test"
PUSH_TO_HUB = False

# LoRA Config
r = 8
lora_alpha = 32
lora_dropout = 0.05
layers = [7]
target_modules = [
    f"model.layers.{i}.{module_name}"
    for i in layers
    for module_name in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
]
task_type = TaskType.CAUSAL_LM

# Training
"""
dtype:
    precision `torch.bfloat16` somehow breaks the
    symmetric property of the P matrix in
    ```
    x @ deltaP.T = x @ (weight_B @ weight_A @ lora_P).T * scaling
                 = x @ lora_P.T @ weight_A.T @ weight_B.T * scaling
    ```
"""
dtype = torch.float32
lr = 3e-05
num_epochs = 100

SAVE_TO_DIRECTORY = False
SAVE_DIRECTORY = "./lora_test"

# %%

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)

# %%


def get_dummy_P_map(target_modules, model):
    P_map = {}
    for name, module in model.named_modules():
        if name in target_modules:
            if isinstance(module, nn.Linear):
                out_features, in_features = module.weight.shape
                dtype = module.weight.dtype
                device = module.weight.device
                P = torch.eye(
                    n=in_features, dtype=dtype, device=device
                )  # Dummy P matrix
                P_map[name] = P
            else:
                raise ValueError(
                    f"Module {name} is not nn.Linear, please implement P for it."
                )
    return P_map


P_map = get_dummy_P_map(target_modules, model)

# %%

# lora_config = LoraConfig(
#     task_type=task_type,
#     r=r,
#     target_modules=target_modules,
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
# )
# peft_model = get_peft_model(model=model, peft_config=lora_config)

lora_config = NullSpaceLoraConfig(
    task_type=task_type,
    r=r,
    target_modules=target_modules,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
)
peft_model = get_peft_model(model=model, peft_config=lora_config)
peft_model.set_lora_P_map(lora_P_map=P_map, adapter_name="default")

print(peft_model)
peft_model.print_trainable_parameters()

# %%

print([k for k in peft_model.state_dict() if "lora_P" in k])

# %%

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = peft_model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))

# %%

messages_list = [
    [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
        {
            "role": "assistant",
            "content": "I am not a pirate chatbot",
        },
    ],
]


class DummyDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, messages_list: list[list[dict]]
    ):
        self.examples = [
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, return_tensors="pt"
            ).squeeze(0)
            for messages in messages_list
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx].clone()}


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}


dataset = DummyDataset(tokenizer, messages_list)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# %%

peft_model.train()
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = peft_model(**batch)
        regularization = sum(
            [
                (delta_weight.norm() ** 2)
                for delta_weight in peft_model.get_delta_weights(
                    adapter="default"
                ).values()
            ]
        )
        loss = outputs.loss + regularization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"{epoch=}, {loss.item()=}, {regularization.item()=}")

# %%

with torch.no_grad():
    peft_model.eval()
    lin = peft_model.base_model.model.model.layers[7].self_attn.q_proj
    x = torch.randn(
        2, lin.in_features, dtype=lin.weight.dtype, device=next(lin.parameters()).device
    )
    y1 = (
        lin.base_layer(x)
        + (lin.lora_B["default"](lin.lora_A["default"](x @ lin.lora_P["default"])))
        * lin.scaling["default"]
    )
    y2 = lin(x)

    deltaP = lin.get_delta_weight("default")

    print(
        f"lora_P symmetric: {torch.allclose(lin.lora_P['default'], lin.lora_P['default'].T)}"
    )
    print(f"lora_A bias is None: {lin.lora_A['default'].bias is None}")
    print(f"lora_B bias is None: {lin.lora_B['default'].bias is None}")
    print(
        f"(y2 - (lin.base_layer(x) + x @ deltaP.T)).norm(): {(y2 - (lin.base_layer(x) + x @ deltaP.T)).norm()}"
    )

    assert torch.allclose(y1, y2, atol=1e-6)
    # Effective additive weight is deltaP; forward adds x @ deltaP.T
    assert torch.allclose(y2, lin.base_layer(x) + x @ deltaP.T, atol=1e-6)

# %%

print(
    f"{peft_model.state_dict()['base_model.model.model.layers.7.self_attn.q_proj.lora_P.default']=}"
)

# %%


def test(model, tokenizer):
    model.eval()
    with torch.no_grad():
        messages_list = [
            [
                {
                    "role": "system",
                    "content": "You are a pirate chatbot who always responds in pirate speak!",
                },
                {"role": "user", "content": "Who are you?"},
            ],
            [
                {
                    "role": "system",
                    "content": "You are a helpful asssistant.",
                },
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]

        for messages in messages_list:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1] :]
            print(tokenizer.decode(response, skip_special_tokens=True))


# %%

test(model=peft_model, tokenizer=tokenizer)

# %%

if SAVE_TO_DIRECTORY:
    peft_model.save_pretrained(save_directory=SAVE_DIRECTORY)

# %%

peft_model = PeftModel.from_pretrained(
    model=model,
    model_id=SAVE_DIRECTORY,
)

# %%

print([k for k in peft_model.state_dict() if "lora_P" in k])
print(
    f"{peft_model.state_dict()['base_model.model.model.layers.7.self_attn.q_proj.lora_P.default']=}"
)

# %%

test(model=peft_model, tokenizer=tokenizer)

# %%

if PUSH_TO_HUB:
    peft_model.push_to_hub(peft_model_id)

# %%

peft_model = PeftModel.from_pretrained(
    model=model,
    model_id=peft_model_id,
)

# %%

test(model=peft_model, tokenizer=tokenizer)

# %%
