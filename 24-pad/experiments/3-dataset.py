# %%

from typing import TypedDict

from datasets import Dataset

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

# %%

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%

class SampleDict(TypedDict):
    question: str
    steps: list[str]
    answer: str
    idx: int

dataset = Dataset.from_dict({
    "question": ["1Hello", "2Hello"],
    "steps": [
        ["Step 11", "Step 12"],
        ["Step 21", "Step 22"]
    ],
    "answer": ["Answer 1", "Answer 2"],
    "idx": [0, 1]
})

print(dataset)

# %% 

dataset.map(
    lambda x: {
        "question_tokenized": x["question"],
        "steps_tokenized": x["steps"],
        "answer_tokenized": x["answer"],
        "idx": x["idx"]
    },
    batched=True,
    remove_columns=["question", "steps", "answer"]
)

# %%

tokenizer(dataset[0]["question"], return_attention_mask=False)

# %%

def tokenize_sample(sample: SampleDict):
    return {
        "question_tokenized": tokenizer(
            sample["question"], 
            return_attention_mask=False
        )["input_ids"],
        "steps_tokenized": tokenizer(
            sample["steps"],
            return_attention_mask=False,
        )["input_ids"],
        "answer_tokenized": tokenizer(
            sample["answer"],
            return_attention_mask=False,
        )["input_ids"],
        "idx": sample["idx"],
    }

procesed_dataset = dataset.map(
    tokenize_sample,
    batched=True,
    remove_columns=["question", "steps", "answer"],
)

procesed_dataset

# %%

dataloader = DataLoader(procesed_dataset, batch_size=2)

# %%

for step, batch in enumerate(dataloader):
    print(step, type(batch["question_tokenized"]))

    # model(**batch)
    # torch.tensor(batch["question_tokenized"])
    print(batch["question_tokenized"])

# %%
