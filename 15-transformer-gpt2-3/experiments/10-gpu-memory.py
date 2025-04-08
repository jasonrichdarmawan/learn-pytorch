# %%

# GPT-2 124 million parameters
batch = 64
vocab_size = 50257
n_embd = 768
n_head = 12
block_size = 2048
n_layer = 12

MAIN = "__main__" == __name__

# %%
# Change me and/or run me again to see the results

n_layer = 24
n_embd = 1536

if MAIN:
    num_params = estimate_parameters_total(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer
    )
    estimated_vram_usage_total = estimate_vram_usage_total(
        batch=batch,
        n_head=n_head,
        block_size=block_size,
        n_embd=n_embd,
        num_params=num_params
    )
    print(f"Number of parameters: {num_params:,}")
    print(f"Estimated VRAM usage total: {estimated_vram_usage_total:.2f} GB")

# %%

def estimate_parameters_word_embeddings(vocab_size: int, n_embd: int) -> int:
    """
    Calculate the number of parameters in the word embeddings.
    :param vocab_size: Vocabulary size
    :param n_embd: Embedding size
    :return: Number of parameters in the word embeddings
    """
    # Word embeddings have a matrix of size (vocab_size, n_embd)
    return vocab_size * n_embd

if MAIN:
    estimate_parameters_word_embeddings(vocab_size=vocab_size, n_embd=n_embd)

# %%

def estimate_parameters_positional_encoding(block_size: int, n_embd: int) -> int:
    """
    Calculate the number of parameters in the positional encoding.
    :param block_size: Block size
    :param n_embd: Embedding size
    :return: Number of parameters in the positional encoding
    """
    # Positional encoding has a matrix of size (block_size, n_embd)
    return block_size * n_embd

if MAIN:
    estimate_parameters_positional_encoding(block_size=block_size, n_embd=n_embd)

# %%

def estimate_parameters_transformer_layer(n_embd: int) -> int:
    self_attention = n_embd * n_embd * 4
    # query, key, value, output projection matrices
    ffn = n_embd * 4 * n_embd * 2
    # feed forward network (ffn)
    # two linear layers (expands to 4x the embedding size, then contracts back)
    layer_norm = n_embd * 2
    # scale and bias terms
    final_layer_norm = n_embd * 2

    return self_attention + ffn + layer_norm + final_layer_norm

if MAIN:
    estimate_parameters_transformer_layer(n_embd=n_embd) * 12

# &&

def estimate_parameters_total(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_layer: int
) -> int:
    """
    Calculate the total number of parameters in the model.
    :param vocab_size: Vocabulary size
    :param block_size: Block size
    :param n_embd: Embedding size
    :param n_layer: Number of transformer layers
    :return: Total number of parameters in the model
    """
    return (
        estimate_parameters_word_embeddings(vocab_size, n_embd) +
        estimate_parameters_positional_encoding(block_size, n_embd) +
        estimate_parameters_transformer_layer(n_embd) * n_layer
    )

if MAIN:
    num_params = estimate_parameters_total(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer
    )

# %%

def estimate_vram_usage_attention_scores(batch: int, n_head: int, block_size: int) -> float:
    """
    Calculate the estimated VRAM usage for attention scores.
    :param batch: Batch size
    :param n_head: Number of attention heads
    :param block_size: Block size
    :return: Estimated VRAM usage in gigabytes
    """
    # Each attention head has a score matrix of size (batch, n_head, block_size, block_size)
    # Each score is a float32 (4 bytes)
    return batch * n_head * block_size * block_size * 4 / 1e9

if MAIN:
    estimate_vram_usage_attention_scores(batch=batch, n_head=n_head, block_size=block_size)

# %%

def estimate_vram_usage_ffn(batch: int, block_size: int, n_embd: int) -> float:
    """
    Calculate the estimated VRAM usage for feedforward networks (FFN).
    :param batch: Batch size
    :param block_size: Block size
    :param n_embd: Embedding size
    :return: Estimated VRAM usage in gigabytes
    """
    # Each FFN has two linear layers with sizes (batch, block_size, n_embd) and (batch, n_embd, block_size)
    # GPT-2 FFN expands the embedding size to 4 times the original size
    # Each weight is a float32 (4 bytes)
    return batch * block_size * (4 * n_embd) * 4 / 1e9

if MAIN:
    estimate_vram_usage_ffn(batch=batch, block_size=block_size, n_embd=n_embd)

# %%

def estimate_vram_usage_parameter(num_params: int) -> float:
    """
    Calculate the estimated VRAM usage for model parameters.
    :param num_params: Number of parameters
    :return: Estimated VRAM usage in gigabytes
    """
    # Each parameter is a float32 (4 bytes)
    return num_params * 4 / 1e9

if MAIN:
    estimate_vram_usage_parameter(num_params=num_params)

# %%

def estimate_vram_usage_gradient(num_params: int) -> float:
    """
    Calculate the estimated VRAM usage for gradients.
    :param num_params: Number of parameters
    :return: Estimated VRAM usage in gigabytes
    """
    # Each gradient is a float32 (4 bytes)
    return num_params * 4 / 1e9

if MAIN:
    estimate_vram_usage_gradient(num_params=num_params)

# %%

def estimate_vram_usage_optimizer(num_params: int) -> float:
    """
    Calculate the estimated VRAM usage for optimizer states.
    :param num_params: Number of parameters
    :return: Estimated VRAM usage in gigabytes
    """
    # Each optimizer state is a float32 (4 bytes)
    # Assuming Adam optimizer, we have 2 states (momentum and variance)
    return num_params * 2 * 4 / 1e9

if MAIN:
    estimate_vram_usage_optimizer(num_params=num_params)

# %%

def estimate_vram_usage_total(
    batch: int,
    n_head: int,
    block_size: int,
    n_embd: int,
    num_params: int
) -> float:
    """
    Calculate the total estimated VRAM usage.
    :param batch: Batch size
    :param n_head: Number of attention heads
    :param block_size: Block size
    :param n_embd: Embedding size
    :param num_params: Number of parameters
    :return: Total estimated VRAM usage in gigabytes
    """
    return (
        estimate_vram_usage_attention_scores(batch, n_head, block_size) +
        estimate_vram_usage_ffn(batch, block_size, n_embd) +
        estimate_vram_usage_parameter(num_params) +
        estimate_vram_usage_gradient(num_params) +
        estimate_vram_usage_optimizer(num_params)
    ) * n_layer

if MAIN:
    estimate_vram_usage_total(
        batch=batch,
        n_head=n_head,
        block_size=block_size,
        n_embd=n_embd,
        num_params=num_params
    )

# %%
