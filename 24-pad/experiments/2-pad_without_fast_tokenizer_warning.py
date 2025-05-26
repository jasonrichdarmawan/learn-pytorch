# %%
"""
In this example:
1.  We load a "slow" tokenizer.
2.  We define `features` as a list of dictionaries, where each dictionary contains `input_ids` and `attention_mask` for a single sequence. This is similar to `non_label_position_features` in your code.
3.  We then call `pad_without_fast_tokenizer_warning` with different padding strategies:
    *   `padding='longest'` (or `padding=True`): Pads all sequences in the batch to the length of the longest sequence.
    *   `padding='max_length'`: Pads all sequences to the specified `max_length`. If a sequence is longer, it would typically be truncated (though `pad_without_fast_tokenizer_warning` itself focuses on padding; truncation is usually handled during the initial tokenization step or by the data collator).
    *   `pad_to_multiple_of`: Ensures the padded length is a multiple of the given number, which can be useful for hardware optimizations.
4.  The function returns a dictionary (or a BatchEncoding object if `return_tensors` is not set) containing the padded `input_ids`, `attention_mask`, and potentially other keys like `token_type_ids` if they were present in the input features and handled by the tokenizer's padding logic.
"""

from transformers import AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import torch

# Load a slow tokenizer (use_fast=False)
# For demonstration, we'll use bert-base-uncased.
# Most modern tokenizers will default to fast if available.
# If a tokenizer doesn't have a fast version, use_fast=False is implicit.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

# Example tokenized features (list of dictionaries)
# This is what your input would look like after tokenizing individual samples
# and before they are batched together.
features = [
    {
        "input_ids": [101, 2054, 2003, 1996, 2034, 2362, 1012, 102], 
        "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]
    }, # "this is the first sentence."
    {
        "input_ids": [101, 2023, 2003, 1037, 2598, 2117, 2362, 1012, 102], 
        "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
    }, # "this is a longer second sentence."
    {
        "input_ids": [101, 2619, 2028, 102], 
        "attention_mask": [1,1,1,1]
    } # "short one."
]

print("Original features:")
for i, feature in enumerate(features):
    print(f"Feature {i}:")
    print(f"  Input IDs: {feature['input_ids']}")
    print(f"  Decoded: {tokenizer.decode(feature['input_ids'])}")
    print(f"  Attention Mask: {feature['attention_mask']}")

# Using pad_without_fast_tokenizer_warning
# Pad to the longest sequence in the batch
batch_longest = pad_without_fast_tokenizer_warning(
    tokenizer,
    features,
    padding='longest', # or True
    return_tensors="pt" # Return PyTorch tensors
)

print("\nBatch padded to longest:")
print("Input IDs:\n", batch_longest['input_ids'])
print("Attention Mask:\n", batch_longest['attention_mask'])
print("Decoded padded inputs (longest):")
for input_ids in batch_longest['input_ids']:
    print(tokenizer.decode(input_ids.tolist()))


# Pad to a specific max_length
max_len = 15
batch_max_length = pad_without_fast_tokenizer_warning(
    tokenizer,
    features,
    padding='max_length',
    max_length=max_len,
    return_tensors="pt"
)

print(f"\nBatch padded to max_length {max_len}:")
print("Input IDs:\n", batch_max_length['input_ids'])
print("Attention Mask:\n", batch_max_length['attention_mask'])
print(f"Decoded padded inputs (max_length {max_len}):")
for input_ids in batch_max_length['input_ids']:
    print(tokenizer.decode(input_ids.tolist()))


# Pad to a multiple of a certain number (e.g., 8 for tensor core efficiency)
pad_mult_of = 8
batch_multiple_of = pad_without_fast_tokenizer_warning(
    tokenizer,
    features,
    padding='longest', # First pad to longest
    pad_to_multiple_of=pad_mult_of,
    return_tensors="pt"
)
print(f"\nBatch padded to a multiple of {pad_mult_of} (after padding to longest):")
print("Input IDs:\n", batch_multiple_of['input_ids'])
print("Attention Mask:\n", batch_multiple_of['attention_mask'])
print(f"Decoded padded inputs (multiple of {pad_mult_of}):")
for input_ids in batch_multiple_of['input_ids']:
    print(tokenizer.decode(input_ids.tolist()))
print(f"Shape of input_ids: {batch_multiple_of['input_ids'].shape}")

# Example with only input_ids (attention_mask will be inferred if not present)
features_only_input_ids = [
    {"input_ids": tokenizer.encode("Hello world", add_special_tokens=True)},
    {"input_ids": tokenizer.encode("A shorter sequence.", add_special_tokens=True)},
]

print("\nOriginal features (only input_ids):")
for i, feature in enumerate(features_only_input_ids):
    print(f"Feature {i}:")
    print(f"  Input IDs: {feature['input_ids']}")
    print(f"  Decoded: {tokenizer.decode(feature['input_ids'])}")


batch_inferred_mask = pad_without_fast_tokenizer_warning(
    tokenizer,
    features_only_input_ids,
    padding=True, # Equivalent to 'longest'
    return_tensors="pt"
)

print("\nBatch with inferred attention mask:")
print("Input IDs:\n", batch_inferred_mask['input_ids'])
print("Attention Mask (inferred):\n", batch_inferred_mask['attention_mask'])
print("Decoded padded inputs (inferred mask):")
for input_ids in batch_inferred_mask['input_ids']:
    print(tokenizer.decode(input_ids.tolist()))
# %%
