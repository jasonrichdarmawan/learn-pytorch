# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval() # Ensure model is in evaluation mode (disables dropout, etc.)

    # Full input sequence
    input_str = "The quick brown fox"
    inputs_pt = tokenizer(input_str, return_tensors="pt")
    input_ids_full = inputs_pt.input_ids
    attention_mask_full = inputs_pt.attention_mask # Shape: (batch_size, seq_length)

    # Get full embeddings
    with torch.no_grad():
        full_embeddings = model.get_input_embeddings()(input_ids_full) # Shape: (batch_size, seq_length, hidden_size)
    
    # Create position_ids for the full sequence. These are absolute positions [0, 1, ..., seq_length-1]
    seq_length = input_ids_full.shape[1]
    position_ids_full = torch.arange(0, seq_length, dtype=torch.long, device=input_ids_full.device).unsqueeze(0) # Shape: (batch_size, seq_length)


    # --- Simulating KV Cache Scenario ---
    tokens_full = tokenizer.convert_ids_to_tokens(input_ids_full[0])
    print(f"Full tokens: {tokens_full}")
    print(f"Full input_ids: {input_ids_full}")
    print(f"Sequence length: {seq_length}")

    # Define the split point. For "The quick brown fox" (4 tokens), idx_split = 2 means:
    # Pass 1: "The quick" (tokens 0, 1)
    # Pass 2: " brown fox" (tokens 2, 3)
    idx_split = 2 
    if idx_split >= seq_length or idx_split <= 0:
        print(f"Warning: idx_split ({idx_split}) is not suitable for sequence length ({seq_length}). Adjusting for demonstration.")
        if seq_length > 1:
            idx_split = seq_length // 2
        else: # Cannot split if seq_length is 0 or 1 in a meaningful way for this demo
             print("Sequence too short to demonstrate split. KV cache demo might not be meaningful.")
             # Fallback to a state that allows code to run, though comparison might be trivial
             idx_split = seq_length 


    # --- Pass 1: Process the first part of the sequence ---
    print("\n--- Pass 1: Processing initial tokens ---")
    # Inputs for the first part of the sequence
    current_input_ids_p1 = input_ids_full[:, :idx_split]
    current_embeddings_p1 = full_embeddings[:, :idx_split]
    current_attention_mask_p1 = attention_mask_full[:, :idx_split] # Attention mask for the first part
    current_position_ids_p1 = position_ids_full[:, :idx_split]     # Absolute positions for the first part [0, 1, ..., idx_split-1]

    print(f"Input for Pass 1 (tokens): {tokenizer.decode(current_input_ids_p1[0]) if current_input_ids_p1.shape[1] > 0 else '[]'}")
    print(f"Input embeddings shape for Pass 1: {current_embeddings_p1.shape}")
    print(f"Attention mask for Pass 1: {current_attention_mask_p1}")
    print(f"Position IDs for Pass 1: {current_position_ids_p1}")

    kv_cache_from_p1 = None
    outputs_p1_logits = None
    if current_input_ids_p1.shape[1] > 0 : # Only run if there are tokens in pass 1
        with torch.no_grad():
            outputs_p1 = model(
                inputs_embeds=current_embeddings_p1,
                attention_mask=current_attention_mask_p1,
                position_ids=current_position_ids_p1,
                use_cache=True # Crucial for generating KV cache
            )
        kv_cache_from_p1 = outputs_p1.past_key_values
        outputs_p1_logits = outputs_p1.logits
        print(f"Logits shape from Pass 1: {outputs_p1_logits.shape}")
        if kv_cache_from_p1:
            print(f"KV cache generated. Length of KV cache tuple (num_layers): {len(kv_cache_from_p1)}")
            print(f"Shape of K from first layer of KV cache: {kv_cache_from_p1[0][0].shape}")


    # --- Pass 2: Process the next part using KV cache from Pass 1 ---
    print("\n--- Pass 2: Processing subsequent tokens with KV cache ---")
    
    past_len = idx_split # Number of tokens processed in Pass 1
    
    # Tokens for Pass 2 are from idx_split to the end of the sequence
    current_input_ids_p2 = input_ids_full[:, past_len:seq_length]
    current_embeddings_p2 = full_embeddings[:, past_len:seq_length, :] # Embeddings for the new tokens only

    # Position IDs for Pass 2 must be absolute positions for the new tokens.
    # E.g., if past_len=2, new tokens start at position 2, 3, ...
    current_position_ids_p2 = position_ids_full[:, past_len:seq_length] 

    # Attention mask for Pass 2 must cover all tokens the model "sees": past (via KV cache) + current (new inputs).
    # So, it's the attention_mask_full up to the current total length.
    combined_attention_mask_p2 = attention_mask_full[:, :seq_length]

    print(f"Input for Pass 2 (tokens): {tokenizer.decode(current_input_ids_p2[0]) if current_input_ids_p2.shape[1] > 0 else '[]'}")
    print(f"Input embeddings shape for Pass 2: {current_embeddings_p2.shape}")
    if kv_cache_from_p1:
        print(f"Past KV cache K tensor shape (1st layer): {kv_cache_from_p1[0][0].shape}")
    else:
        print("Past KV cache is None (e.g. if Pass 1 was empty).")
    print(f"Attention mask for Pass 2 (covers past+current): {combined_attention_mask_p2}")
    print(f"Shape of attention_mask for Pass 2: {combined_attention_mask_p2.shape}")
    print(f"Position IDs for Pass 2 (absolute for current tokens): {current_position_ids_p2}")

    outputs_p2_logits = None
    if current_input_ids_p2.shape[1] > 0 : # Only run if there are new tokens to process
        if kv_cache_from_p1 is None and past_len > 0:
             print("Error: kv_cache_from_p1 is None, but past_len > 0. This indicates an issue in Pass 1 processing or logic.")
             # This case should ideally not happen if past_len > 0 implies Pass 1 ran and produced cache.
        
        with torch.no_grad():
            outputs_p2 = model(
                inputs_embeds=current_embeddings_p2,
                attention_mask=combined_attention_mask_p2, 
                position_ids=current_position_ids_p2,     
                past_key_values=kv_cache_from_p1, # Use the cache from Pass 1
                use_cache=True # Typically True if further generation is needed, can be False if only logits for this pass are needed
            )
        outputs_p2_logits = outputs_p2.logits
        print(f"Logits shape from Pass 2: {outputs_p2_logits.shape}")
    elif past_len > 0 and kv_cache_from_p1 is not None : # No new tokens, but past cache exists. Predict next token based on Pass 1.
        # This is like generating the token *after* the ones in Pass 1.
        # We need at least one "dummy" input token to prompt the next prediction if the model requires it,
        # or check if the model can produce a logit from just past_key_values (uncommon for inputs_embeds route).
        # For simplicity, this demo focuses on when Pass 2 has actual tokens.
        # If current_input_ids_p2 is empty, outputs_p2.logits would typically be empty too.
        # The comparison logic later handles empty outputs_p2_logits.
        print("Pass 2 has no new tokens. Logits from Pass 2 will be empty or correspond to a next token prediction if model supports it with empty input.")


    # --- Verification: Full pass without KV cache ---
    print("\n--- Verification: Full pass without KV cache ---")
    outputs_full_logits = None
    if input_ids_full.shape[1] > 0:
        with torch.no_grad():
            outputs_full = model(
                inputs_embeds=full_embeddings,
                attention_mask=attention_mask_full,
                position_ids=position_ids_full
            )
        outputs_full_logits = outputs_full.logits
        print(f"Logits shape from full pass: {outputs_full_logits.shape}")

    # Comparison
    # We compare the logits for the last token of the sequence.
    # If Pass 2 processed tokens [T_split, ..., T_end], its last logit is for T_end.
    # The full pass also has a logit for T_end at its last position.
    can_compare = (
        outputs_p2_logits is not None and outputs_p2_logits.shape[1] > 0 and
        outputs_full_logits is not None and outputs_full_logits.shape[1] > 0 and
        # Ensure that Pass 2 actually processed the segment that includes the last token of the full sequence.
        # This is true if past_len + outputs_p2_logits.shape[1] == seq_length
        (past_len + outputs_p2_logits.shape[1] == seq_length)
    )

    if can_compare:
        logits_p2_last = outputs_p2_logits[0, -1]    # Logits for the last token from Pass 2 output
        logits_full_last = outputs_full_logits[0, -1] # Logits for the last token from full pass output

        print(f"Comparing last token's logits (token index {seq_length - 1}):")
        print(f"torch.allclose: ", torch.allclose(logits_p2_last, logits_full_last, atol=1e-5)) # Check if they are close within tolerance
        
        print(f"Logits from P2 (last token, first 5 values): {logits_p2_last[:5]}")
        print(f"Logits from Full (last token, first 5 values): {logits_full_last[:5]}")

    else:
        print("Could not perform comparison. This might be due to empty sequences in one of the passes, or idx_split configuration.")
        if outputs_p2_logits is not None and outputs_full_logits is not None:
             print(f"Debug comparison info: past_len={past_len}, p2_logit_len={outputs_p2_logits.shape[1] if outputs_p2_logits is not None else -1}, seq_len={seq_length}")


if __name__ == "__main__":
    main()

# %%