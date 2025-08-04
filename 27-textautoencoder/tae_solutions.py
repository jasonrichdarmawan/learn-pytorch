""" Exercises on Text Autoencoders"""
# %% [markdown]
# # Section 1: Text Autoencoders - Exploring SONAR

# This notebook explores Meta's SONAR text autoencoder, which can encode text
# into fixed-size vectors and decode them back to (approximately) the original text.

# Learning objectives:
# 1. Load and use SONAR for text encoding/decoding
# 2. Understand the properties of text embeddings
# 3. Test robustness to noise
# 4. Explore how text length affects embeddings
# 5. Experiment with token swapping and sentence combinations

# %% [markdown]
# ## Setup and Installation
#
# First, we need to install SONAR and its dependencies. Just run, nothing worth reading here unless you get errors.
# Note: You may need to adjust the CUDA version in fairseq2 installation.

# %%
# !pip install -q fairseq2==0.4.5 sonar-space==0.4.0 torchvision==0.21.0 torch==2.6.0 torchaudio==2.6.0 plotly nbformat numpy>=2.0.0 jaxtyping
# !pip install -q -U datasets

import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import json
from jaxtyping import Float

# Check if CUDA is available
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
torch.set_grad_enabled(False)  # We're only doing inference
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Loading SONAR Models
#
# SONAR (Sentence-Level Multimodal and Language-Agnostic Representations) is Meta's text autoencoder
# that can encode entire sentences/paragraphs into fixed-size vectors and decode them back to approximately
# the original text.
#
# **What are Text Autoencoders?**
#
# Text Autoencoders are models that compress entire input sequences (sentences/paragraphs) into a single
# fixed-size vector representation (the "bottleneck"), then reconstruct the original text from that vector.
# Unlike typical text embedding models that only encode, these models have both an encoder AND decoder.
#
# ![Text Autoencoder Architecture](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/db8d350884974ce6dcb1281011c5053e11b65711c12a4556.png)
#
# **How Text Autoencoders Work:**
# 1. **Encoder**: Takes input text → processes through Transformer → outputs single fixed-size vector (1024-dim)
# 2. **Bottleneck**: The compressed representation that captures semantic meaning in a dense vector
# 3. **Decoder**: Takes the vector → generates text that approximates the original input
#
# **Key Properties:**
# - **Lossy compression**: Some information is lost, but semantic meaning is preserved
# - **Fixed-size representation**: Any length text becomes same-size vector (useful for comparison/clustering)
# - **Cross-lingual**: Can encode in one language and decode in another
# - **Reconstruction capability**: Unlike embedding-only models, you can decode back to text
# - **Semantic preservation**: The bottleneck captures core meaning even with compression
#
# **SONAR Specifically:**
# - Trained on ~100B tokens with denoising and translation objectives
# - Uses 24-layer Transformer encoder and decoder, with mean-pooling to create the bottleneck vector
# - Supports 200+ languages and can handle up to 512 tokens of context
# - Currently one of the best-performing text autoencoders available
#

# %% [markdown]
# We start by loading the models.
print("Loading SONAR models...")
text2vec = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE,
)
vec2text = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE,
)
print("Models loaded successfully!")

# %% [markdown]
# ## Basic Usage - Encoding and Decoding
#
# Test basic encoding and decoding functionality.

# %%
# Simple example sentences
sentences = [
  'My name is SONAR.',
  'I can embed sentences into vectorial space.'
]

# Encode sentences to vectors
embeddings = text2vec.predict(sentences, source_lang="eng_Latn")
print(f"Embeddings shape: {embeddings.shape}")  # Should be [2, 1024]
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"L2 norm of embeddings: {torch.norm(embeddings, dim=1).tolist()}")

# Decode vectors back to text
reconstructed = vec2text.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
print("\nReconstruction quality:")
for orig, rec in zip(sentences, reconstructed):
    print(f"Original:      {orig}")
    print(f"Reconstructed: {rec}")
    print()

"""
Personal note:
1. Why are we interested with `torch.norm(embeddings, dim=1)`

   We are interested in `torch.norm(embeddings,dim=1)`
   because it computes the L2 norm (magnitude)
   of each embedding vector produced by the
   text autoencoder. This tells us:
   - How "large" or "strong" each embedding is in
     vector space.
   - Whether the model produces embeddings with
     consistent magnitude for different texts.
   - If there is a relationship between the length
     of content of the text and the embedding's norm
   - Whether the embedding space is normalized
     or has biases (e.g., longer texts might
     have larger norms)

   Analyzing the L2 norm helps us understand the
   geometry and scaling of the embedding space,
   which is important for tasks like similarity
   comparison, clustering, and robustness analysis

2. What is geometry? How do we understand the geometry
   and scaling of the embedding space from the L2 norm?
   
   In this context, geometry refers to the structure
   and arrangement of the embedding vectors
   in high-dimensinoal space, how they are distributed,
   how far apart they are, and how their directions
   and magnitudes relate to each other

   By examing the L2 norm (magnitude) of embeddings,
   we gain insight into:
   - Scaling: Whether embeddings are normalized
     (all have similar length) or if some are
     much larger than others
   - Distribution: If longer or more complex texts
     produce embeddings with larger norms,
     it suggests that space is not normalized
     and the model encodes more information
     as magnitude.
   - Biases: If certain types of texts (e.g., 
     longer repetitive, or random) consistently
     have higher or lower norms, it reveals biases
     in how the model represents information
   - Similarity: In a normalized space, cosine
     similarity is more meaningful. If norms
     vary widely, Euclidean distance and cosine
     similarity may behave differently

   So, by analyzing the L2 norms, we understand
   how the model uses the embedding space to represent
   different texts, and whether the space is well-behaved
   for downstream tasks like clustering or
   similarity search
"""

# %% [markdown]
# ## Exercise 1: Testing with Longer, More Realistic Text
# Let's test how well SONAR handles paragraph-length text.
#
# Write a function to reconstruct text from SONAR embeddings, and try testing with some longer text.

def reconstruct_text(texts: list[str]) -> list[str]:
    """Reconstruct text from SONAR embedding, by first encoding and then decoding the text.

    Args:
        texts: List of strings to embed and then reconstruct.

    Returns:
        List of reconstructed strings.
    """
    # [your implementation here]
    embedding = text2vec.predict(texts, source_lang="eng_Latn")
    print(f"Embeddings shape: {embedding.shape}")  
    print(f"Embedding dimension: {embedding.shape[1]}")
    print(f"L2 norm of embeddings: {torch.norm(embedding, dim=1).tolist()}")
    return vec2text.predict(embedding, target_lang="eng_Latn", max_seq_len=512)

# Longer example paragraphs
paragraph1 = """SONAR is a model from August 2023, trained as a semantic text auto-encoder,
converting text into semantic embed vectors, which can later be decoded back into text.
Additionally, the model is trained such that the semantic embed vectors are to some degree
"universal" for different languages, and one can embed in French and decode in English."""

paragraph2 = """I tried it, and SONAR seems to work surprisingly well. For example, the above
paragraph and this paragraph, if each are encoded into two 1024 dimensional vectors
(one for each paragraph), the model returns the following decoded outputs."""

paragraph3 = """\
Your text here.\
"""

# Test with paragraphs
long_texts = [paragraph1, paragraph2, paragraph3]
long_reconstructed = reconstruct_text(long_texts)

print("Paragraph reconstruction:")
max_print_length = 400
for i, (orig, rec) in enumerate(zip(long_texts, long_reconstructed)):
    print(f"\n--- Paragraph {i+1} ---")
    print(f"Original ({len(orig)} chars):")
    print(orig[:max_print_length] + "..." if len(orig) > max_print_length else orig)
    print(f"\nReconstructed ({len(rec)} chars):")
    print(rec[:max_print_length] + "..." if len(rec) > max_print_length else rec)

# %% [markdown]
# How well does it work for longer text? It should be doing a pretty good job. Bonus: How long does the text get before you see some degradation?

# %% [markdown]
# ## Exercise 2: Noise Robustness Analysis
#
# In this exercise, we investigate SONAR's robustness to perturbations in the embedding space.
# We'll systematically add Gaussian noise of increasing magnitude to text embeddings and analyze
# how reconstruction quality degrades. This helps us understand:
# 1. How stable the embedding space is to small perturbations
# 2. The sensitivity of the decoder to different noise directions
#
# Write a function to test the robustness of SONAR to noise, and try it out with some different noise levels.

"""
Personal note:
1. Why we are interested with noise robustness
   analysis? In what instance we want to add
   gaussian noise to text embeddings in practical
   use case? Is this strictly for analysis
   or there is a practical use case?

   We are interested in noise robustness analysis
   because it helps us understand how stable
   and reliable the text autoencoder's embedding
   space is. By adding Gaussian noise to embeddings
   and observing how the decoded text changes,
   we can:
   - Assess the stability of the embedding:
     Does small noise cause large changes in meaning?
   - Evaluate the robustness of the decoder:
     Can it recover the original text from
     slightly perturbed embeddings?
   - Identify sensitive directions in the embedding
     space
   
   Practical use case for adding Gaussian noise
   to text embeddings include:
   - Data augmentation: Slightly noised embeddings
     can be used to train more robust downstream
     models
   - Privacy: Adding noise can help anonymize
     or obfuscate sensitive information in embeddings
   - Adversarial robustness: Testing how
     models behave under perturbations can reveal
     vulnerabilities
   - Compression and transmission: In lossy
     environments (e.g., low-bandwidth communication),
     embeddings may be corrupted; robustness
     ensures graceful degradation

   While noise robustness analysis is often used
   for analysis and research, it also has practical
   implications for building reliable, robust,
   and privacy-preserving NLP systems.
"""

"""
Personal note:
1. About the data augmentation, why do we want 
   slightly noised embeddings to train more robust 
   downstream models? Is the downstream models 
   refers to instruct model where it accepts 
   tokenized text as inputs or is the downstream 
   models accept embeddings as inputs?

   We add slight noise to embeddings during training
   to make downstream models more robust to small
   variations or errors in the input
   - Downstream models here usually refers
     to models that accept embeddings as inputs
     (e.g., classifiers, clustering algorithms,
     retrieval systems, or other neural networks
     that operate on embeddings)
   - For models that accept tokenized text (like
     instruct models), noise augmentation is typically
     done at the text level (e.g., paraphrasing,
     word swaps), not on embeddings

2. About the privacy, how adding noise can help 
   anonymize or obfuscate sensitive information 
   in embeddings?

   Adding noise to embeddings can help
   anonymize or obfuscate sensitive information because:
   - It makes it harder to reconstruct the
     original text or infer private details
     from the embedding
   - Even if someone obtains the embedding, the
     added noise reduces the risk of
     extracting sensitive content or identifying
     individuals
   - This is similar to differential privacy,
     where noise is added to protect user data

3. About adversarial robustness, what practical 
   use case where an attacker can exploit the 
   vulnerability? Is there a concept where the 
   attack have access to the embeddings and 
   thus can do perturbations or perturbation can 
   also happen at text level?

   - Practical attack scenario: If an attacker can
     access the embeddings (e.g., in a retrieval
     or recommendation system), they might
     craft small perturbations to the embeddings to
     fool the downstream model (e.g., to bypass
     filters, trigger wrong recommendations,
     or extract information)
   - Attack surface:
     - If the attacker can only access the text
       interface, they can still craft adversarial
       texts that, when encoded, produce embeddings
       that fool the system
     - If the attacker has access to the
       embeddings directly (e.g., in API-based
       systems or shared embedding spaces),
       they can manipualte embeddings
       more precisely
   - Perturbations can happen at both the text
     and embedding level, but direct embedding
     attacks require more access

In summary:
- Data augmentation with noise is for embedding-
  based downstream models
- Noise for privacy makes it harder to extract
  sensitive info from embeddings
- Adversarial attacks can target both text
  and embedding levels, depending on
  attack access     
"""

def test_noise_robustness(text, noise_levels):
    """Test how reconstruction quality degrades with noise.

    """
    # Get original embedding
    original_emb = text2vec.predict([text], source_lang="eng_Latn")
    original_norm = torch.norm(original_emb)

    print(f"Embeddings shape: {original_emb.shape}")  
    print(f"Embedding dimension: {original_emb.shape[1]}")
    print(f"L2 norm of embeddings: {original_norm.tolist()}")

    results = []
    for noise_scale in noise_levels:
        # [your implementation here]
        # Add Gaussian noise
        noise = torch.randn_like(original_emb)
        noise = noise_scale * original_norm * noise / torch.norm(noise)
        noisy_emb = original_emb + noise

        # Decode noisy embedding
        reconstructed = vec2text.predict(noisy_emb, target_lang="eng_Latn", max_seq_len=512)[0]

        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_emb, noisy_emb, dim=1
        ).item()

        results.append({
            'noise_scale': noise_scale,
            'cosine_similarity': cosine_sim,
            'reconstruction': reconstructed
        })

    return results

# Test with different noise levels
test_text = "The quick brown fox jumps over the lazy dog."
# test_text = """SONAR is a model from August 2023, trained as a semantic text auto-encoder,
# converting text into semantic embed vectors, which can later be decoded back into text.
# Additionally, the model is trained such that the semantic embed vectors are to some degree
# "universal" for different languages, and one can embed in French and decode in English."""

noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"Original text: {test_text}\n")
results = test_noise_robustness(test_text, noise_levels)

for res in results:
    print(f"Noise scale: {res['noise_scale']:.1f}")
    print(f"Cosine similarity: {res['cosine_similarity']:.3f}")
    print(f"Reconstructed: {res['reconstruction']}")
    print()

"""
Personal note:
1. why the noise is computed as noise = noise_scale 
   * original_norm * noise? my understanding 
   is projection of vector B onto vector A is 
   (vector A @ vector B) / vector A magnitude * 
   vector A / vector A magnitude. 
   so what is original_norm * noise / noise 
   magnitude trying to compute?

   ```
   noise = noise_scale * original_norm * noise / torch.norm(noise)
   ```

   What is it doing?
   - `noise` is a random vector (from `torch.randn_like(original_emb)`),
     so its direction is random but its magnitude is arbitrary
   - `torch.norm(noise)` computes the L2 norm
     (magnitude) of this random noise vector
   - `noise / torch.norm(noise)` normalizes
     the noise vector to have unit length (L2 norm = 1)
   - `original_norm * ...` scales the unit noise
     vector to have the same magnitude
     as the original embedding
   - `noise_scale * ...` further scales the
     noise by the desired amount (e.g., 0.1, 1.0, etc)

   So, `noise = noise_scale * original_norm * noise
   / torch.norm(noise)` creates a vector that:
   - Has the same direction as the random noise,
   - Has a magnitude = noise_scale * original_norm

   This means the noise is proportional to the size
   of the original embedding, and the `noise_scale`
   controls how much noise you add (relative to
   the embedding's norm)

   How is this different from projection?
   - Projection is about finding the component
     of one vector along the direction of another
   - Here, we are not projecting the noise onto
     the embedding
   - Instead, we are scaling the random noise
     so that its magnitude is a certain fraction
     (or multiple) of the embedding's magnitude

Summary:
This approach ensures that the amount of noise
added is meaningful relative to the scale of
the embedding, regardless of the absolute values
of the random noise vector. It is not a projection;
it's a normalization and scaling operation
"""

# %% [markdown]
# What do you see?
# It should be the case that with little noise, the reconstruction is still good. With more noise, the reconstruction gets worse. However, I found there is a lot of variance in the results, so try running it a few times. It seems like some directions have basically no effect, and others have a lot of effect.

# %% [markdown]
# ## Exercise 3: Text Length vs Vector Norm Analysis
#
# ### Exercise 3: Investigating the Relationship Between Text Length and Embedding Norms
#
# In this exercise, we'll explore whether there's a correlation between the length of text
# and the L2 norm (magnitude) of its embedding vector. This analysis will help us understand:
# - How semantic information is distributed across embedding dimensions
# - Whether longer texts result in larger embedding magnitudes
# - If the embedding space has inherent biases based on text length
#
# We'll test this hypothesis using three different types of text:
# 1. Repeated words (to test pure length effects)
# 2. Random character sequences (to test meaningless content)
# 3. Natural language sentences (to test realistic content)

# %%
import plotly.express as px
import pandas as pd
import random
import string

# Collect all data first
data = []
def add_data(text, text_type):
  emb = text2vec.predict([text], source_lang="eng_Latn")
  norm = torch.norm(emb).item()
  data.append({
    'text': text,
    'length': len(text),
    'norm': norm,
    'type': text_type
  })

# Repeated words (more examples)
for length in range(1, 100):
  for word in [
    'word', 'sentence', 'paragraph', 
    'dog', 'spicy', 'anime',
  ]:
    words = [word] * length
    text = ' '.join(words)
    add_data(text, 'Repeated Words')

# Random characters (more examples)
random.seed(42)
for length in range(1, 100, ):
  random_words = [
    ''.join(
        random.choices(
          string.ascii_lowercase, 
          k=random.randint(3, 8)
        )
      ) for _ in range(length)
  ]
  text = ' '.join(random_words)
  add_data(text, 'Random Characters')

# Normal sentences (many more examples)
normal_sentences = [
  "Hi",
  "Hello",
  "Good morning",
  "Hello there",
  "How are you?",
  "Nice to meet you",
  "The cat sat on the mat",
  "I like to read books",
  "The weather is nice today",
  "She went to the store yesterday",
  "The quick brown fox jumps over the lazy dog",
  "I enjoy listening to music in the evening",
  "She sells seashells by the seashore on weekends",
  "To be or not to be, that is the question",
  "The early bird catches the worm every morning",
  "A picture is worth a thousand words in most cases"
]
for text in normal_sentences:
  add_data(text, 'Real Text')

# Load dataset of some example texts generated by Llama3b
dataset = load_dataset("nickypro/fineweb-llama3b-regen-split", split="train")
for split_text in dataset.select(range(20)):
  for paragraph in split_text['split_text']:
    add_data(paragraph, 'Real Text')


# Create DataFrame and plot
df = pd.DataFrame(data)
# Truncate text to first 50 characters for hover display
df['text_truncated'] = df['text'].str[:50] + '...'
fig = px.scatter(df,
  x='length', y='norm', color='type',
  title="Text Length vs Embedding Norm",
  labels={
    'length': 'Text Length (characters)', 
    'norm': 'Embedding L2 Norm'
  },
  hover_data=['text_truncated'],
  opacity=0.5,
  log_x=True,
)

fig.show()

# %% [markdown]
# ## Exercise 4: Token Swapping Experiments
#
# This exercise explores how we can manipulate text embeddings to perform token swapping.
# We'll investigate:
# 1. Building difference vectors between similar texts
# 2. Applying global transformations to swap words
# 3. Creating position-specific transformations for targeted edits

# %%
# Helper functions

def diff_vector(
  src_text: str, 
  tgt_text: str
) -> Float[torch.Tensor, "1024"]:
  """Return embedding difference between *tgt_text* and *src_text* (tgt − src)."""
  # [your implementation here]
  src_emb = text2vec.predict(
    [src_text], 
    source_lang="eng_Latn"
  )
  tgt_emb = text2vec.predict(
    [tgt_text], 
    source_lang="eng_Latn",
  )
  return (tgt_emb - src_emb).squeeze(0)

def decode(
  embedding: torch.Tensor, 
  max_seq_len: int = 512
) -> str:
  """Greedy‑decode a single 1024‑D embedding back to text."""
  # [your implementation here]
  return vec2text.predict(
    embedding.unsqueeze(0), 
    target_lang="eng_Latn", 
    max_seq_len=max_seq_len,
  )[0]


def positional_diff(
  src_word: str, 
  tgt_word: str, 
  pos: int, *, 
  seq_len: int, 
  filler: str = "_",
) -> torch.Tensor:
  """Build a difference vector that swaps **src_word→tgt_word** at index *pos*.

  All other positions are filled with *filler* tokens so that the vector is
  specific to that location.
  """
  # [your implementation here]
  src_tokens = [filler] * seq_len
  tgt_tokens = [filler] * seq_len
  src_tokens[pos] = src_word
  tgt_tokens[pos] = tgt_word
  return diff_vector(
    src_text=" ".join(src_tokens), 
    tgt_text=" ".join(tgt_tokens),
  )

assert diff_vector("dog", "cat").shape == (1024,)
assert isinstance(decode(torch.randn(1024), 5), str)
assert positional_diff("dog", "cat", pos=1, seq_len=8, filler="a").shape == (1024,)

# %% [markdown]
# Now we can try see what the difference vector does in different cases.

# 1. Global dog→cat vector
print("1. Global word swapping:")
swap_vec = diff_vector("dog", "cat")
sentence = "the dog is happy in the dog house"
sent_emb = text2vec.predict(
  [sentence], 
  source_lang="eng_Latn"
).squeeze(0)

print(f"Original:               {decode(sent_emb)}")
print(f"Global swap dog→cat:    {decode(sent_emb + swap_vec)}")

"""
Personal note:
1. why to swap dog→cat we don't scale the swap_vec 
   with the original embedding norm, the same way 
   we scale the gaussian noise in the exercise 2: 
   noise robustness analysis?

   1. Gaussian Noise (Exercise 2: Noise Robustness)
      - Goal: Add random perturbations to the
        embedding to test how robust the decoder is to
        noise
      - Why scale by the original embedding norm?
        - The magnitude of the embedding can
          vary depending on the input text
        - Scaling the noise by the embedding's norm
          ensures that the noise is proportional
          to the "size" of the embedding, making
          the effect of noise comparable across
          different inputs.
        - This lets you interpret `noise_scale=1.0`
          as "noise with the same magnitude as
          the embedding itself"
    2. Word Swap Vector (dog→cat)
       - Goal: Apply a semantic transformation
         in embedding space that consistently
         swaps "dog" for "cat"
       - Why NOT scale by the original embedding
         norm?
         - The swap vector `swap_vec = 
           embedding("cat") - embedding("dog")`
           is a fixed direction and magnitude
           in embedding space, representing
           the semantic difference between "dog"
           and "cat"
         - Adding this vector to any sentence
           embedding (E.g., "the dog is happy")
           is intended to shift the meaning
           from "dog" to "cat" without regard
           to the sentence's original norm
         - Scaling by the sentence's norm
           would distort the intended semantic
           shift–the swap would be too strong
           for long sentences and too weak
           for short ones, and would not
           represent the true "dog→cat"
           transformation

Summary:
- Noise: Scaled by norm to keep perturbation
  size meaningful relative to the embedding
- Semantic swap: Fixed vector, not scaled, because
  the semantic difference is independent of
  the sentence's embedding norm

This ensures that semantic edits (like word swaps)
are consistent, while noise is always relative
to the embedding's scale
"""

# 2. Position‑specific swap
print("\n2. Position-specific swapping:")
# Swap only the token at index 1 (0‑based) in a sentence
pos_vec = positional_diff("dog", "cat", pos=1, seq_len=8, filler="a")
print(f"Position‑aware swap:    {decode(sent_emb + pos_vec)}")

# 3. Test with different word pairs
print("\n3. Testing different word pairs:")
word_pairs = [
  ("happy", "sad"), 
  ("house", "home"), 
  ("big", "small"),
]
for src, tgt in word_pairs:
  swap_vec = diff_vector(src, tgt)
  test_sentence = f"the {src} animal lives here"
  test_emb = text2vec.predict(
    [test_sentence], 
    source_lang="eng_Latn"
  ).squeeze(0)
  print(f"{src}→{tgt}: '{test_sentence}' → '{decode(test_emb + swap_vec)}'")

# %% [markdown]
# ## Exercise 5: Sentence Combination
#
# This exercise explores how we can combine two sentences into a single embedding.
# So far I have only tried a couple of the most naive approaches. It's ok but I suspect it should be easy to try better approaches to this also.

# %% [markdown]
# ### Part 1: Basic Combination Analysis
#
# First, let's analyze how SONAR combines sentences with different relationships.

# %%
# Create diverse sentence pairs for analysis
sentence_pairs = [
  # Related sentences (continuation)
  ("Related sentences (continuation)", "The weather is beautiful today", "I think I'll go for a walk"),
  ("Related sentences (continuation)", "She opened the mysterious letter", "Her hands trembled as she read it"),

  # Contrasting sentences
  ("Contrasting sentences", "I love sunny days", "But I hate the rain"),
  ("Contrasting sentences", "The movie was exciting", "However, the ending disappointed me"),

  # Unrelated sentences
  ("Unrelated sentences", "Cats are independent animals", "Python is a programming language"),
  ("Unrelated sentences", "The Earth orbits the Sun", "Pizza is my favorite food"),

  # Question-answer pairs
  ("Question-answer pairs", "What's your favorite color?", "My favorite color is blue"),
  ("Question-answer pairs", "Where do you live?", "I live in New York City"),
]

# Analyze combinations
combination_data = []
for label, sent_a, sent_b in sentence_pairs:
  # Individual embeddings
  emb_a = text2vec.predict(
    [sent_a], 
    source_lang="eng_Latn"
  )
  emb_b = text2vec.predict(
    [sent_b], 
    source_lang="eng_Latn"
  )

  # Combined embeddings (both orders)
  combined_ab = f"{sent_a} {sent_b}"
  combined_ba = f"{sent_b} {sent_a}"
  emb_ab = text2vec.predict(
    [combined_ab], 
    source_lang="eng_Latn"
  )
  emb_ba = text2vec.predict(
    [combined_ba], 
    source_lang="eng_Latn"
  )

  # Various combinations
  emb_avg = (emb_a + emb_b) / 2
  emb_sum = emb_a + emb_b
  emb_diff = emb_a - emb_b

  # Calculate similarities
  data = {
    'label': label,
    'sent_a': sent_a[:30] + '...' if len(sent_a) > 30 else sent_a,
    'sent_b': sent_b[:30] + '...' if len(sent_b) > 30 else sent_b,
    'sim_ab_a': torch.nn.functional.cosine_similarity(
      emb_ab, emb_a
    ).item(),
    'sim_ab_b': torch.nn.functional.cosine_similarity(
      emb_ab, emb_b
    ).item(),
    'sim_ab_ba': torch.nn.functional.cosine_similarity(
      emb_ab, emb_ba
    ).item(),
    'sim_ab_avg': torch.nn.functional.cosine_similarity(
      emb_ab, emb_avg
    ).item(),
    'sim_ab_sum': torch.nn.functional.cosine_similarity(
      emb_ab, emb_sum
    ).item(),
    'order_sensitivity': torch.norm(
      emb_ab - emb_ba
    ).item()
  }
  combination_data.append(data)

# Display results
df_comb = pd.DataFrame(combination_data)
print("Sentence Combination Analysis:")
print(df_comb.to_string(index=False))

"""
Personal note:
1. about the exercise 5: sentence combination, 
   part 1: basic combination analysis. why do we 
   care about cosine similarity between (emb_ab, 
   emb_a), (emb_ab, emb_b), (emb_ab, emb_ba), 
   (emb_ab, emb_avg), (emb_ab, emb_sum), (emb_ab - 
   emb_ba)? what is order sensitivity? why order 
   sensitivity is calculated as the norm of 
   (emb_ab - emb_ba)? why we do care about the 
   order sensitivity?

   1. Cosine Similarity (emb_ab, emb_a) and (emb_ab, emb_b)
      - What: Measures how similar the embedding
        of the combined sentence (`emb_ab`) is to
        each individual sentence embedding (`emb_a`
        or `emb_b`)
      - Why: If `emb_ab` is very close to `emb_a`
        or `emb_b`, it means the combined
        embedding msotly represents one sentence,
        not both. Ideally, we want `emb_ab`
        to capture information from both
   2. Cosine Similarity (emb_ab, emb_ba)
      - What: Compares the embedding of "A B"
        (`emb_ab`) to "B A" (`emb_ba`)
      - Why: If the similarity is very high,
        the model is order-insensitive (it doesn't
        care about the order of sentences).
        If it's lower, the model
        is order-sensitive (it encodes order information).
        For concatenation, order should matter
   3. Cosine Similarity (emb_ab, emb_avg) and
      (emb_ab, emb_sum)
      - What: compares the true combined embedding
        (`meb_ab`) to simpler linear combinations
        of the individual embeddings (average or sum).
      - Why: This tells us whether a naive combination
        (like averaging or summing) is a good
        approximation for the true embedding of
        the concatenated sentence. If similarity is
        high, simple methods might suffice for
        combining meanings.
   4. Order Sensitivity norm(emb_ab - emb_ba)
      - What: The L2 norm (distance) between
        the embedding of "A B" and "B A"
      - Why: This quantifies how much the
        embedding changes when the order of sentence
        is swapped. If the norm is small,
        the model doesn't encode order well; if it's
        large, the model is sensitive to order
      - Why do we care? In antural language,
        order matters ("The cat chased the dog"
        ≠ "The dog chased the cat"). For tasks
        like concatenation, we want the embedding
        to reflect this difference
  Summary:
  - These metrics help us understand how well the
    embedding space captures both the content
    and the order of combined sentences.
  - Order sensitivity is important because,
    for many applications, the meaning of a
    sentence pair dependes on the order,
    and we want our embeddings to reflect that

2. the order_sensitivity value ranges from 0.14 
   to 0.23, is this small?
   the sim_ab_ba and order_sensitivity contradicts. 
   sometimes when a sentence pair have higher 
   sim_ab_ba than another sentence pair, 
   it doesn't translate to the former have 
   lower order_sensitivity. why is this the case?

   1. Is order_sensitivity (0.14-0.23) small?
      For embeddings with L2 norm around 20-30
      (typical for 1024-dim SONAR), a difference
      of 0.143-0.23 is relatively small. This means
      swapping the order of sentences
      only slightly changes the embedding,
      so the model is not highly sensitivite to
      order–but it does encode some order information
   2. Why do sim_ab and order_sensitivity sometimes
      contradict?
      - `sim_ab_ba` is cosine similarity
        (measures angle between vectors, ignores
         magnitude)
      - `order_sensitivity` is L2 norm (measures
        absolute distance, considers both direction
        and magnitude)
      Two pairs can have:
      - High cosine similarity (vectors point
        in nearly the same direciton) but large
        L2 distance (if their magnitudes differ)
      - Low cosine similarity but small L2 distance
        (if vectors are close in space but point in
        slightly different directions)
"""

# %% [markdown]
# ### Part 2: Try simple linear combination
# If we want to combine two sentences, we can just add their embeddings? Or maybe average them? Will this give us something that works as an embedding with two sentences side-by-side?

# %%
class SimpleLinearCombiner(nn.Module):
  def __init__(self, embed_dim=1024):
    super().__init__()
    self.embed_dim = embed_dim

  def forward(self, x, y):
    return x + y

basic_combiner_model = SimpleLinearCombiner().to(DEVICE)

# Test the simple linear combiner
def test_performance_on_new_examples(model, verbose=True):
  """Test model performance on predefined pairs plus one random example"""
  model.eval()

  # Predefined test pairs
  test_pairs = [
    ("It started raining heavily.", "Everyone ran for shelter."),
    ("First, preheat the oven.", "Then, mix the ingredients."),
    ("The book was fascinating.", "The movie adaptation was terrible."),
    ("I need to buy milk.", "I also need to get bread."),
  ]

  # Add one random pair
  # idx1, idx2 = np.random.choice(len(all_sentences), 2, replace=False)
  # test_pairs.append((all_sentences[idx1], all_sentences[idx2]))

  test_results = []

  for sent1, sent2 in test_pairs:
    # Get embeddings
    emb1 = text2vec.predict(
      [sent1], 
      source_lang="eng_Latn"
    ).to(DEVICE)
    emb2 = text2vec.predict(
      [sent2], source_lang="eng_Latn"
    ).to(DEVICE)
    emb_true = text2vec.predict(
      [f"{sent1} {sent2}"], 
      source_lang="eng_Latn"
    ).to(DEVICE)

    # Predict and decode
    with torch.no_grad():
      emb_pred = model(emb1, emb2)

    text_true = vec2text.predict(
      emb_true.cpu(), 
      target_lang="eng_Latn"
    )[0]
    text_pred = vec2text.predict(
      emb_pred.cpu(), 
      target_lang="eng_Latn"
    )[0]
    similarity = torch.cosine_similarity(
      emb_pred, 
      emb_true, 
      dim=-1
    ).item()

    test_results.append({
      'sent1': sent1, 
      'sent2': sent2, 
      'decoded_true': text_true,
      'decoded_pred': text_pred, 
      'similarity': similarity
    })

    if verbose:
      print(f"\nSent1: {sent1}")
      print(f"Sent2: {sent2}")
      print(f"True: {text_true}")
      print(f"Pred: {text_pred}")
      print(f"Similarity: {similarity:.4f}")

  avg_similarity = np.mean(
    [r['similarity'] for r in test_results]
  )
  print(f"\nAverage similarity: {avg_similarity:.4f}")
  return test_results

# Test the simple linear combiner
test_results = test_performance_on_new_examples(basic_combiner_model)

# %% [markdown]
# ### Part 3: Better ways of combining sentences.
#
# We can try to do better than just a simple linear combination to try get behaviour like concatenation. For this, we will need some training data.
#
# We'll create a dataset of sentence pairs and their combined embeddings to train our model.
# As a source of data, use the provided dataset of llama-3.2-3b-instruct generated text.


from datasets import load_dataset
print("Getting training data...")
dataset = load_dataset("nickypro/fineweb-llama3b-regen-split", split="train")
# Extract individual sentences
all_sentences = []
for item in dataset.select(range(100)):  # Use first 100 documents
  for paragraph in item['split_text']:
    # Split paragraph into sentences (simple approach)
    sentences = paragraph.split('. ')
    for sent in sentences:
      if 10 < len(sent) < 200:  # Filter by length
        all_sentences.append(sent.strip())
# Limit to manageable size
all_sentences = all_sentences[:2000]
print(f"Collected {len(all_sentences)} sentences")

# %% [markdown]
# What do you see?
# In general, you should see that this kinda gets a sentence that is the same as one of the original sentences, or inbetween the two sentences. It doesn't really append one sentence to the other.

# %% [markdown]
# ### Part 4: Create Training Data for Sentence Combination
#
# Now we need to create training data to teach our model how to combine sentence embeddings.
# The goal is to learn a function that maps two individual sentence embeddings to the embedding
# of their concatenation.
#
# **Your task**: Create pairs of sentences and compute their embeddings along with the embedding
# of their concatenated form. This will give us input-output pairs for training.
#
# **Steps to implement**:
# 1. Randomly select pairs of sentences from our collected sentences
# 2. Compute embeddings for each individual sentence using SONAR
# 3. Create a concatenated sentence by joining them with a space
# 4. Compute the embedding of the concatenated sentence (this is our target)
# 5. Store all embeddings and original text for training
#
# **Expected outcome**: A dataset where each example contains:
# - Original sentences text for reference
# - Two individual sentence embeddings (inputs)
# - The embedding of their concatenation (target output)

# %%

def create_training_data(
  all_sentences: list[str], 
  n_pairs: int = 1000
) -> list[dict]:
  """Create training data for sentence combination.

  Args:
    all_sentences: List of sentences to create training data from.
    n_pairs: Number of pairs to create.

  Returns:
    List of dictionaries with training data.
    Each dictionary contains:
    - 'sent1': First sentence
    - 'sent2': Second sentence
    - 'emb1': Embedding of the first sentence
    - 'emb2': Embedding of the second sentence
    - 'emb_combined': Embedding of the concatenated sentence
  """
  print("Creating sentence pairs and embeddings...")
  training_data = []

  for i in tqdm(range(n_pairs)):
    # Randomly select two sentences
    # [your implementation here]
    idx1, idx2 = np.random.choice(
      len(all_sentences), 
      2, 
      replace=False,
    )
    sent1, sent2 = all_sentences[idx1], all_sentences[idx2]

    # Compute embeddings
    emb1 = text2vec.predict(
      [sent1], 
      source_lang="eng_Latn"
    )
    emb2 = text2vec.predict(
      [sent2], 
      source_lang="eng_Latn"
    )

    # Compute combined embedding
    combined = f"{sent1} {sent2}"
    emb_combined = text2vec.predict(
      [combined], 
      source_lang="eng_Latn"
    )
    # [~# end of exercise]

    training_data.append({
      'sent1': sent1,
      'sent2': sent2,
      'emb1': emb1.cpu(),
      'emb2': emb2.cpu(),
      'emb_combined': emb_combined.cpu(),
    })

  print(f"Generated {len(training_data)} training examples")
  return training_data

training_data = create_training_data(all_sentences)

# %% [markdown]
# ### Part 5: Trained scale combination model
#
# Now let's create a more sophisticated model that learns how to combine two sentence embeddings.
# This model will have learnable parameters that can be optimized to better concatenate sentences.
#
# **Exercise**: Implement a ScaleCombinerModel that learns optimal weights for combining embeddings:
# - Initialize learnable scale parameters for each input embedding
# - Add a learnable constant bias term
# - The output should be: const + scale1*embedding1 + scale2*embedding2

# %% [markdown]
# define the simple scaled linear combiner model

class ScaleCombinerModel(nn.Module):
  """
  Simple linear combiner model:
  output = const + (scale1)*x + (scale2)*y
  """
  def __init__(self, embed_dim=1024):
    super().__init__()
    self.embed_dim = embed_dim

    # Constant bias
    self.const = nn.Parameter(
      torch.zeros(embed_dim)
    )

    # Scalar weights for original embeddings
    self.scale1 = nn.Parameter(
      torch.ones(1) * 0.5
    )
    self.scale2 = nn.Parameter(
      torch.ones(1) * 0.5
    )

  def forward(self, x, y):
    # Simple linear combination
    # [your implementation here]
    output = self.const + self.scale1 * x + self.scale2 * y
    return output

# Initialize model
scale_combiner_model = ScaleCombinerModel(
  embed_dim=1024
).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in scale_combiner_model.parameters()):,}")

# %% [markdown]
# Write the training loop for the model.
#

# %%

def combined_cosine_norm_loss(pred, target):
  cos_sim = F.cosine_similarity(pred, target, dim=-1)
  cosine_loss = (1 - cos_sim).mean()
  norm_loss = torch.abs(pred.norm(dim=-1) - target.norm(dim=-1)).mean()
  loss = cosine_loss + norm_loss
  return loss

class CombinerModelTrainer:
  """Trainer class for the ScaleCombinerModel."""

  def __init__(self, model, device=None):
    self.model = model
    self.device = device or torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
    )
    self.train_losses = []
    self.test_losses = []

  def prepare_data(
    self, 
    training_data, 
    test_size=0.2, 
    random_state=42
  ):
    """Prepare training data by stacking embeddings and splitting train/test."""
    X1 = torch.stack(
      [d['emb1'].squeeze(0) for d in training_data]
    )
    X2 = torch.stack([
      d['emb2'].squeeze(0) for d in training_data]
    )
    Y = torch.stack(
      [
        d['emb_combined'].squeeze(0) 
        for d in training_data
      ]
    )

    # Split into train/test
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
      X1, X2, Y, test_size=test_size, 
      random_state=random_state
    )

    # Convert to tensors and move to device
    self.X1_train = X1_train.to(self.device)
    self.X2_train = X2_train.to(self.device)
    self.Y_train = Y_train.to(self.device)
    self.X1_test = X1_test.to(self.device)
    self.X2_test = X2_test.to(self.device)
    self.Y_test = Y_test.to(self.device)

  def train_epoch(
    self, optimizer, criterion, batch_size=32,
  ):
    """Train for one epoch."""
    self.model.train()
    epoch_loss = 0

    steps = ( len(self.X1_train) + (batch_size - 1) ) // batch_size

    for i in range(steps):
      # [your implementation here]
      batch_x1 = self.X1_train[i:i+batch_size]
      batch_x2 = self.X2_train[i:i+batch_size]
      batch_y = self.Y_train[i:i+batch_size]

      optimizer.zero_grad()
      pred = self.model(batch_x1, batch_x2)
      loss = criterion(pred, batch_y)
      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), 
        max_norm=1.0, 
        norm_type=2,
      )
      optimizer.step()

      epoch_loss += loss.item() / steps

    return epoch_loss, norm

  def evaluate(self, criterion):
    """Evaluate model on train and test sets."""
    self.model.eval()
    with torch.no_grad():
      # [your implementation here]
      test_pred = self.model(self.X1_test, self.X2_test)
      test_loss = criterion(test_pred, self.Y_test).item()

    return test_loss

  def train(
    self, training_data, epochs=100, lr=1e-3, 
    batch_size=32, verbose=True,
  ):
    """Train the combiner model on the provided training data."""
    # Prepare data
    self.prepare_data(training_data)

    # Training setup
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = combined_cosine_norm_loss

    # Reset loss tracking
    self.train_losses = []
    self.test_losses = []

    if verbose:
      print("Training a combiner model...")

    for epoch in range(epochs):
      # Training
      epoch_loss, norm = self.train_epoch(optimizer, criterion, batch_size)

      # Evaluation
      test_loss = self.evaluate(criterion)

      self.train_losses.append(epoch_loss)
      self.test_losses.append(test_loss)

      if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
        print(f"Epoch {epoch+1}: Epoch Loss = {epoch_loss:.4f}, Norm: {norm:.4f}, Test Loss = {test_loss:.4f}")

    return self.train_losses, self.test_losses

# Train the model
try:
  torch.set_grad_enabled(True)  # We're now training but only in this cell
  trainer = CombinerModelTrainer(scale_combiner_model, DEVICE)
  train_losses, test_losses = trainer.train(training_data)
# except Exception as e:
#   print(f"Error training model: {e}")
#   if hasattr(e, 'traceback'):
#     print(e.traceback)
finally:
  torch.set_grad_enabled(False)  #

# %% [markdown]
# ### Part 6: Test Performance on New Examples

# %%
print("\nModel Performance on Test Examples:")
print("=" * 80)

test_results = test_performance_on_new_examples(scale_combiner_model)

# %% [markdown]
# What do you see?
# It does a better job, it seems to be 
# approximately one sentence followed 
# by the other, but kind still mixes 
# the two sentences up a but sometimes.

# %% [markdown]
# ## Bonus Exercise: Try to improve the model.
# Maybe there are better ways to combine the sentences to get concat? Can you get it so that it reliably concatenates two sentences in the correct order?
# %%
class SelfAttention(nn.Module):
  def __init__(self, embed_dim, n_head):
    super().__init__()
    self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
    self.c_proj = nn.Linear(embed_dim, embed_dim)
    self.attn_dropout = nn.Dropout(0.1)
    self.resid_dropout = nn.Dropout(0.1)
    self.embed_dim = embed_dim
    self.n_head = n_head

    self.register_buffer(
      "bias", 
      torch.tril(
        torch.ones(embed_dim, embed_dim)
      ).view(1, 1, embed_dim, embed_dim)
    )
  
  def forward(self, x):
    B, T, C = x.size()

    q, k, v = self.c_attn(x).split(self.embed_dim, dim=-1)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

    att = (q @ k.transpose(-2, -1)) * (C // self.n_head)**-0.5
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
    att = att.softmax(dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1, 2).reshape(B, T, C)

    y = self.c_proj(y)
    y = self.resid_dropout(y)

    return y

class Block(nn.Module):
  def __init__(self, embed_dim, n_head):
    super().__init__()
    self.ln1 = nn.LayerNorm(embed_dim)
    self.attn = SelfAttention(embed_dim, n_head=n_head)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.mlp = nn.Sequential(
      nn.Linear(embed_dim, embed_dim * 2),
      nn.GELU(),
      nn.Linear(embed_dim * 2, embed_dim),
      nn.Dropout(0.1),
    )
  
  def forward(self, x1, x2):
    combined = torch.stack([x1, x2], dim=1)
    # x = combined + self.attn(self.ln1(combined))
    # x = x + self.mlp(self.ln2(x))
    x = combined + self.mlp(self.ln2(combined))
    return x.unbind(dim=1)

class BetterCombinerModel(nn.Module):
  """
  Simple linear combiner model:
  output = const + (scale1)*x + (scale2)*y
  """
  def __init__(self, embed_dim=1024, n_layer=2):
    super().__init__()
    self.embed_dim = embed_dim
    # Constant bias
    self.const = nn.Parameter(torch.zeros(embed_dim))
    # other parameters
    # [your code here]
    self.n_layer = n_layer

    self.h = nn.ModuleList([
      Block(embed_dim, n_head=16) 
      for _ in range(n_layer)
    ])
    self.lm_head = nn.Linear(embed_dim, embed_dim, bias=False)

    self.apply(self.__init_weights)

    # Scalar weights for original embeddings
    self.scale1 = nn.Parameter(
      torch.ones(1) 
      * 2**-0.5
    )
    self.scale2 = nn.Parameter(
      torch.ones(1) 
     * 2**-0.5
    )

  def __init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = module.out_features**-0.5
      std *= (2 * self.n_layer)**-0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)

  def forward(
    self, 
    x1: Float[torch.Tensor, "batch_size embed_dim"], 
    x2: Float[torch.Tensor, "batch_size embed_dim"]
  ):
    """
    What I've tried:
    1. concat + Linear; very bad performance
    2. MLP; very bad performance
    3. MLP + residual stream; doesn't increase performance
    3. Dropout; doesn't increase performance
    4. Self-attention; doesn't increase performance
    """
    # Best Test Loss=0.3583
    output = self.const + self.scale1 * x1 + self.scale2 * x2
    # [your implementation here]
    # for block in self.h:
    #   x1, x2 = block(x1, x2)
    # output = x1 + x2
    return output


# Initialize model
better_combiner_model = BetterCombinerModel(
  embed_dim=1024
).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in better_combiner_model.parameters()):,}")

# Train the model
try:
  torch.set_grad_enabled(True)  # We're now training but only in this cell
  trainer = CombinerModelTrainer(better_combiner_model, DEVICE)
  train_losses, test_losses = trainer.train(training_data, epochs=1000, lr=6e-4)
# except Exception as e:
#   print(f"Error training model: {e}")
#   if hasattr(e, 'traceback'):
#       print(e.traceback)
finally:
  torch.set_grad_enabled(False)  #

 # %%
print("\nModel Performance on Test Examples:")
print("=" * 80)

test_results = test_performance_on_new_examples(better_combiner_model)

# %%