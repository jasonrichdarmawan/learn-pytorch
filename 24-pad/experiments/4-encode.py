# %%

from transformers import AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

# %%

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# %%

batch = [
    "Hello",
    "Hello Hello"
]

question_tokenized = (
    tokenizer(batch + ["\n"], 
              add_special_tokens=True,
              return_attention_mask=False,
              padding=True,
              return_tensors="pt",)
)

print(type(question_tokenized))

print(type(question_tokenized["input_ids"]))

# %%

print(tokenizer.encode("Hello", add_special_tokens=True))

# %%

pad_without_fast_tokenizer_warning(
    tokenizer,
    [
        {
            "input_ids": tokenizer.encode("Hello", 
                                          add_special_tokens=True), 
            "attention_mask": [1]
        },
        {
            "input_ids": tokenizer.encode("Hello Hello", 
                                          add_special_tokens=True),
            "attention_mask": [0, 0]
        }
    ],
    return_tensors="pt",
)

# %%
