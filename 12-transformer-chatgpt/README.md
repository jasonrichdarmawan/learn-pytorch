# Tokenizer

1. [Tokenization](https://www.youtube.com/watch?v=fNxaJsNG3-s)
2. [Turning sentences into data](https://www.youtube.com/watch?v=r9QjkdSJZ2g)
3. [TOkenizers](https://www.youtube.com/watch?v=hL4ZnAWSyuU)

## Out of vocabulary problem

train_data = "I love my dog"
test_data = "I really love my dog"

test_data_tokenized = "I", "love", "my", "dog" (in numbers)

Solution? Add `<OOV>` to represent out of vocabulary / never seen token. This helps maintain the sequence length to be the same length as the sentence.

test_datA_tokenzed = "I", "<OOV>", my, "dog"

## Different sentence length problem.

sentences = [
    "I love my dog",
    "Do you think my dog is amazing?"
]

Solution? Add padding (in number, it's `0`) 

train_data_tokenized = [
    [ "I", "love",    "my", "dog", "pad", "pad",     "pad"],
    ["Do",  "you", "think",  "my", "dog",  "is", "amazing"]
]

# GPU Resources problem

1. [1.58bit](https://www.youtube.com/watch?v=wCDGiys-nLA)

# Hallucination problem

1. [Chain-of-Verification (COVE) method](https://www.youtube.com/watch?v=Lar3K2gN454)

# Encoder-Decoder, Encoder-Only, Decoder-Only

Transfromer has 3 versions:
1. Encoder-Decoder transformer, used in text-to-text (like translating text), text-to-image (like image generation), image-to-text (like image captioning).
2. Encoder-Only transformer, used in generating context-aware embeddings (For example, the user inputs `The pizza came out of the oven and it tasted good!` because the transformer use Self-Attention instead of Masked Self-Attention, the transformer can correctly associate the token `it` with the token `pizza` instead of with the token `oven`). This context-aware embeddings can help cluster similar sentences or even similar documents.
3. Decoder-Only transformer, used in text-to-text (like predicting the next token. For example, the user inputs `what is statquest`, the GPT will respond `awesome`).

# QKV

For example, `What is StatQuest`

Q: `is`
K: `What`, `is`
V: `What`, `is`

Q * K^T: obviously Query `is` is more similar with Key `is`

Let's say
Softmax(Q * K^T): 0.0 for `What`, 1.0 for `is`

Softmax(Q * K^T) * V: 0.0 * Value `What` + 1.0 * Value `is`

So, the attention of `is` is the calculation above.

# Be careful of `torch.nn.CrossEntropyLoss`

```
criterion = torch.nn.CrossEntropyLoss()
# expected input: N, C, d1
# expected output: N, d1
loss = criterion(input, target)
```