# %%

from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example sentences
sentences = [
    "This is the first sentence.",
    "This is a longer second sentence."
]

# Tokenize the sentences
encoded_inputs = tokenizer(sentences, padding=False, truncation=False) # Turn off automatic padding initially

print("Encoded inputs before padding:")
for i, input_ids in enumerate(encoded_inputs['input_ids']):
    print(f"Sentence {i+1}: {input_ids}")
    print(f"Decoded: {tokenizer.decode(input_ids)}")

# Pad the tokenized inputs
# The `pad` method can take a dictionary of encoded inputs
# or a list of lists (input_ids)
padded_inputs = tokenizer.pad(
    encoded_inputs,
    padding='longest',  # Pad to the longest sequence in the batch
    return_tensors="pt"  # Return PyTorch tensors
)

print("\nEncoded inputs after padding:")
print(padded_inputs['input_ids'])

print("\nDecoded padded inputs:")
for input_ids in padded_inputs['input_ids']:
    print(tokenizer.decode(input_ids))

# You can also pad to a specific length:
padded_to_max_length = tokenizer.pad(
    encoded_inputs,
    padding='max_length',
    max_length=20, # Specify the desired maximum length
    return_tensors="pt"
)
print("\nEncoded inputs after padding to max_length=20:")
print(padded_to_max_length['input_ids'])

print("\nDecoded padded inputs (max_length=20):")
for input_ids in padded_to_max_length['input_ids']:
    print(tokenizer.decode(input_ids))
