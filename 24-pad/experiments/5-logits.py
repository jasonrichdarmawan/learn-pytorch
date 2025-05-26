# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%

def main():
    # 1. Load the model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Example input string
    input_text = "1 2 3 4 5 6 7 8 9 10"
    print(f"Input text: '{input_text}'")

    # 3. Tokenize the input
    # `return_tensors="pt"` returns PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Tokenized input_ids: {input_ids}")
    print(f"Shape of input_ids: {input_ids.shape}") # (batch_size, sequence_length)

    # 4. Pass the tokenized input to the model
    # We use torch.no_grad() as we are not training, just doing inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # `logits` shape: (batch_size, sequence_length, vocab_size)

    # 5. Print the shape of outputs.logits
    print(f"Shape of outputs.logits: {logits.shape}")

    # 6. Demonstrate getting logits for predictions at each position
    # For "hello world", if tokenized into N tokens:
    # logits[0, 0, :] are scores for token after input_ids[0, 0]
    # logits[0, 1, :] are scores for token after input_ids[0, 1]
    # ...
    # logits[0, N-1, :] are scores for token after input_ids[0, N-1] (i.e., after "world")

    # Logits for the token *after* the last token of the input sequence ("world")
    last_token_logits = logits[0, -1, :]
    print(f"Shape of logits for the token after the last input token: {last_token_logits.shape}")

    # 7. Optionally, decode the most likely next token
    predicted_next_token_id = torch.argmax(last_token_logits).item()
    predicted_next_token = tokenizer.decode(predicted_next_token_id)
    print(f"Predicted next token ID: {predicted_next_token_id}")
    print(f"Predicted next token: '{predicted_next_token}'")

    full_sequence_predictions = []
    for i in range(input_ids.shape[1]):
        predicted_token_id_at_step_i = torch.argmax(logits[0, i, :]).item()
        predicted_token_at_step_i = tokenizer.decode(predicted_token_id_at_step_i)
        input_token_at_step_i = tokenizer.decode(input_ids[0, i].item())
        full_sequence_predictions.append({
            "input_token": input_token_at_step_i,
            "predicted_next_token": predicted_token_at_step_i
        })

    print("\nPredictions at each step of the input sequence:")
    for pred in full_sequence_predictions:
        print(f"  After input token '{pred['input_token']}', predicted next token: '{pred['predicted_next_token']}'")

if __name__ == "__main__":
    main()

# %%
