# %%
# Use Python Interactive Window in VS Code to run this code
# It's similar to Jupyter Notebook, but instead of `.ipynb` files, it uses `.py`

import torch as t
from torch import Tensor
from transformer_lens import FactoredMatrix
from transformer_lens import HookedTransformerConfig
from transformer_lens import HookedTransformer
from transformer_lens import utils
from transformer_lens import ActivationCache
from plotly_utils import imshow
import einops
from jaxtyping import Int
from jaxtyping import Float

from pathlib import Path
import sys

device = t.device(
  "mps" if t.backends.mps.is_available()
  else "cuda" if t.cuda.is_available()
  else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %%

if MAIN:
  cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,
    positional_embedding_type="shortformer",
  )

# %%

if MAIN:
  """
  ```
  WORKSPACE_PATH = "/Users/jason"

  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download callummcdougall/attn_only_2L_half attn_only_2L_half.pth --local-dir "$WORKSPACE_PATH/transformers/callummcdougall/attn_only_2L_half"
  ```
  """
  WORKSPACE_PATH = "/Users/jason"

  weights_path = f"{WORKSPACE_PATH}/transformers/callummcdougall/attn_only_2L_half/attn_only_2L_half.pth"

# %%

if MAIN:
  model = HookedTransformer(cfg)
  pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
  model.load_state_dict(pretrained_weights)

# %%

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens, logits, cache). This
    function should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache

# %%

"""
4. Reverse-engineering induction circuits

Learning Objectives:
- Understand the difference between 
investigating a circuit by looking at activation patterns,
and reverse-engineering a circuit by lookign directly at the weights.
- Use the factored matrix class 
to inspect the QK and OV circuits within an induction circuit
- Perform futher exploration of induction circuits: composition scores,
and targeted ablations

In previous exercises we looked at the attention patterns and attributions
of attention heads to try and and identify which ones were important in the
intduction circuit. This might be a good way to get a feel for the circuit,
but it's not a very rigorous way to understand it. It would be better described
as feature analysis, where we observe that a particular head seems to be
performing some task on certain class of inputs, without identifying why it does so

Now we're going to do some more rigorous mechanistic analysis - digging into
the weights and using them to reverse engineer the induction head algorithm
and verify that it is really doing what we tihnk it is

# Referesher - the induction circuit
Before we get into the meat of this section, let's refresh the results 
we've gotten so far from ivnestigating induction heads. We've found:
- When fed repeated sequences of tokens, heads `1.4` and `1.10` have the charcteristic
  induction head attention pattern of a digonal stripe with offset `seq-len - 1`
  - We saw this both from the CircuitsVis results, and from the fact that these
    heads had high induction scores by our chosen metric (with all other heads
    having much lower scores)
  - We also saw that head `0.7` strongly attends to the previous token in the
    sequence (even on non-repeated sequences).
  - We performed logit attribution on the model, and found that the values
    written to the residual stream by heads `1.4` and `1.10` were both improtant
    for getting us correct predictions in the second half of the sequence.
  - We performed zero-ablation on the model, and found that heads `0.7`, `1.4`,
    and `1.10` all resulted in a large accuracy degradation on the repeated
    sequence task when they were ablated.

Based on all these observations, try and summarise the induction circuit and
how it works, in your own words. You should try and link your explanation to
the QK and OV circuits for particular heads, and describe what type (or types)
of attention head composition are taking place.

You can use the dropdown below to check your understanding.

My summary of the algorithm:
- Head `0.7` is a previous token head (the QK-circuit ensures it always attends
  to the previous token).
- The OV circuit of head `0.7` writes a copy of the previous token in a different
  subspace to the one used by the embedding.
- The OV-circuit of head `1.10` copies the value of the soruce token 
  to the same output logit
  - Note that this iscopyign from the embedding subspace, not the `0.7` output
    subspace - it is not using V-Composition at all
- `1.4` is also performing the same role as `1.10` (so together they can be
  more accurate - we'll see exactly how later).

  To emphasise - the sophisticated hard part is computing the attention pattern
  of the induction head - this takes careful composition. The previous token
  and copying parts are fairly easy. This is a good illustrative example
  of how the QK circuits and OV circuits act semi-independently, and are often
  best thought of somewhat separately. And that computing the attention patterns
  can involve real and sophisticated computation!

  Below is a diagram of the induction circuit, with the heads indicated
  in the weight matrices.
  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described_3.png

  Questions from the image [1]:
  1. Does W_{OV}^{0.7} = W_V^{0.7} \cdot W_O^{0.7}?
  2. If true, then does the "I am "D" W_OV = I follow "D" statement
     is the same as 
     I am "D" \cdot W_V^{0.7} \cdot W_O^{0.7} = I follow "D"?

     However, I am not sure where is I am "D" \cdot W_V^{0.7} \cdot W_O^{0.7}
     is used in the code [2]. Is the I follow "D" the same as y = self.c_proj(y),
     if the destination token only pays attention to the source token
     at the position of the vector "D"?

     What I know:

     If a destination token is supposed to "follow D", the
     attention score will be close to 1.0 for the source token "D"
     and close to 0.0 for all others.

     code from [2]
     ```
     q, k, v = self.c_attn(x).split(self.nembd, dim=2)
     ...
     attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.nembd))
     ...
     y = attn @ v
     ...
     y = self.c_proj(y)
     ```

  Reference:
  [1] https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described_3.png
  [2] https://github.com/jasonrichdarmawan/learn-pytorch/blob/7fa819260a33992b49bcca1e632dbc6562a85f55/15-transformer-gpt2-3/train_gpt2.py#L38-L66

  # Refresher - QK and OV circuits
  Before we start, a brief terminology note. I'll refer to weight marices for
  a particular layer and head using superscript notation, e.g. W_Q^{1.4} is
  the query matrix for the 4th head in layer 1, and it has shape `(d_model, d_head)`
  (remmber that we multiply with weight marices on the right). Similarly,
  attention patterns will be denoted A^{1.4} (remember that these are activations,
  not parameters, since they're given by the formula A^h = xW_{QK}^hx^T, where
  x is the residual stream (with shape `[seq_len, d_model]`).
  
  As a shorthand, I'll often have A denote the one-hot encoding of token `A`
  (i.e., the vector with zeros everywhere except a one at the index of `A`),
  so A^TW_E is the embedding vector for `A`.

  Lastly, I'll refer to special matrix products as follows:
  - W_{OV}^h = W_V^hW_O^h is the OV circuit for head h, and W_EW_{OV}^hW_U is the full OV circuit
  - W_{QK}^h = W_Q^h(W_K^h)^T is the QK circuit for head h, and W_E_W_{QK}^hW_E^T is the full QK circuit
  
  Note that the order of these matrices are slightly different from the Mathematical
  Frameworks paper - this is a consequence of the way TransformerLens stores
  its weight matrices.

  # Question - what is the itnerpretation of each of the following matrices?
  They are quite a lot of questions here, but they are conceptually important.
  If you're confused, you might want to read the answers to the first few questions
  and then try the later ones.
  In your answers, you should describe the type of input it takes, and what the
  outputs represnet.

  W_{OV}^h
  Answer
    W_{OV}^h has size (d_model, d_model), it is a linear map describing what
    information gets moved from source to destination, in the residual stream.

    In other words, if x is a vector in the residual stream, then x^TW_{OV}^h
    is the vector written to the residual stream at the destination position,
    if the destination token only pays attention to the source token at the
    position of the vector x.
  
  W_EW_{OV}^hW_U
  Hint
    if A is the one-hot encoding for token `A` (i.e. the vector with zeros everywhere
    except for a one in the position corresponding to token `A`), then think
    about what A^TW_EW_{OV}^hW_U represents. You can evaluate this expression
    from left to right (e.g. start with thinking about what A^TW_E represents,
    then multiply by the other two matrices).
  Answer
    W_EW_{OV}^hW_U has size (d_vocab, d_vocab), it is a linear map describing
    what information gets moved from source to destination, in a start-to-end
    snese.
    If A is the one-hot encoding for token `A`, then:
    - A^TW_E is the embedding vector for `A`
    - A^TW_EW_{OV}^h is the vector which would get written to the residual
      stream at the destination position, if the destination token only pays
      attention to `A`.
    - A^TW_EW_{OV}^hW_U is the unembedding of this vector, i.e. the thing
      which gets added to the final logits.
  
  W_{QK}^h
  Answer
    W_{QK}^h has size (d_model, d_model), it is a bilinear form describing
    where information is moved to and from in the residual stream (i.e.
    which residual stream vectors attend to which others).
    x_i^TW_{QK}^hx_j = (x_i^TW_Q^h)(x_j^TW_K^h)^T is the attention
    score paid by token i to token j
  
  W_EW_{QK}^hW_E^T
  Answer
    W_EW_{QK}^hW_E^T has size (d_vocab, d_vocab), it is a bilinear form
    describing where information is moved to and from, among words
    in our vocabulary (i.e. which tokens pay attention to which others).

    if A and B are one-hot encodings for tokens `A` and `B`, then
    A^TW_EW_{QK}^hW_E^TB is the attention score paid by token `A` to token `B`:
    A^T_W_EW_{QK}^hW_E^TB = (A^TW_EW_Q^h)(B^TW_EW_K^h)^T
  
  W_posW_{QK}^hW_E^T
  Answer
    W_{pos}W_{QK}^hW_{pos}^T has size (n_{ctx}, n_{ctx}), it is a bilinear
    form describing where information is moved to and from, among tokens
    in our context (i.e. which token positions pay attention to other positions).
    If i and j are one-hot encodings for positions i and j, (in other words,
    they are just the ith and jth basis vectors), then i^TW_{pos}W_{QK}^hW_{pos}^Tj
    is the attention score paid by the position with position i to the position with position j:
    i^TW_{pos}W_{QK}^hW_{pos}^Tj = (i^TW_posW_Q^h)(j^TW_{pos}W_K^h)^T

  W_EW_{OV}^{h_1}W_{QK}^{h_2}W_E^T
  where h_1 is in an earlier layer than h_2
  Hint
    This matrix is best seen as a bilinear form of size (d_vocab, d_vocab).
    The (A, B)-th element is:
    (A^TW_EW_{OV}^{h_1})W_{QK}^{h_2}(B^TW_E)^T
  Answer
    W_EW_{OV}^{h_1}W_{QK}^{h_2}W_E^T has size (d_vocab, d_vocab), it is a
    bilinear form describing where information is moved to and from in head h_2,
    given that the query-side vector is formed from the output of head h_1.
    In other words, this is an instance of Q-composition.
    if A and B are one-hot encoding for tokens `A` and `B`, then
    A^TW_EW_{OV}^{h_1}W_{QK}^{h_2}W_E^TB is the attention score paid to
    token `B`, by any token which attended strongly to an `A`-token in head h_1.

    To further break this down, if it still seems confusing:
    A^TW_EW_{OV}^{h_1}W_{QK}^{h_2}W_E^TB = (A^TW_EW_{OV}^{h_1}W_Q^{h_2})(B^TW_EW_K^{h_2})^T

    Note that the actual attentino scores will be a sum of multiple terms,
    not just this one (in fact, we'd have a different term for every combination
    of query and key input). But this term describes the particular
    contribution to the attention scores form this combination of query
    and key input, and it might be the case that this term is the only
    one that matters (i.e. all other terms don't much affect the final
    probbilities). We'll see soemthing exactly like this later on

    Before we start, there's a problem that we might run into when calculating
    all these matrices. Some of them are massive, and might not fit on our
    GPU. For instance, both full circuit matrices have shape (d_vocab, d_vocab),
    which in our case means 50278 x 50278 = 2.5 billion elements. Even if
    your GPU can handle this, it still seems inefficient. Is there any way
    we can meaningfully analyse these matrices, without actually having
    to calculate them?

    # Facotired Matrix class
    In transformer interpretability, we often need to analyse low rank factorized
    matrices - a matrix M = AB, where M is (large, large), but A is (large, small),
    and B is (small, large). This is a common structure in transformers.

    For instance, we can factorise the OV circuit above as W_{OV}^h = W_V^hW_O^h,
    where W_V^h has shape (768, 64) and W_O^h has shape (64, 768). For an even
    more extreme example, the full OV circuit can be written as (W_EW_V^h)(W_O^hW_U),
    where these two matrices have shape (50278, 64) and (64, 50278)
    respectively. Similarly, we can write the full QK circuit as (W_EW_Q^h)(W_EW_K^h)^T

    The FactoredMatrix class is a convenient way to work with these. It implements
    efficient algorithms for various operations on these, such as computing
    the trace, eigenvalues, Frobenius norm, singular value decomposition,
    and products with other matrices. it can (approximately) act as a drop-in
    replacement for the original matrix.

    This is all possible because knowing the factorisation of a matrix give us
    a much easier way of computng its important properties. Intuitively, since
    M = AB is a very large matrix that operates on very small subspaces, we
    shouldn't expect knowing the actual values M_{ij} to be the most
    efficient way of storing it
"""

# %%
# Exercise - deriving properties of a factored matrix
"""
To give you an idea of what kinds of properties youcan easily compute if you have
a facotred matrix, let's try and derive some ourselves.
Suppose we have M = AB, w here A has shape (m, n), and m > n. So M is a size-(m,m)
matrix with rank at most n.

Qustion - how can you easily compute the trace of M?
Answer
  We have:
  Tr(M) = TR(AB) = \sum_{i=1}^m \sum_{j=1}^n A_{ij}B_{ji}
  so evaluation of the trace is O(mn).
  Note that, by cyclicity of the trace, we can also show that Tr(M) = Tr(BA)
  (although we don't even need to calculate the product AB to evaluate the trace)

Question - how can you easily compute the eigenvalues of M?
(As you'll see in later exercises, eigenvalues are very important for
evaluating matrices, for instance we can assess the copying scores of an OV
circuit by looking at the eigenvalues of W_{OV})
Hint
  It's computationally cheaper to find the eigenvalues of BA rather than AB.
  How are the eigenvalues of AB and BA related?
  How are the eigenvalues of Ab and BA related?
Answer
  The eigenvalues of AB and BA are related as follows: If v is an eigenvector
  of AB with ABv = \lambda v, then Bv is an eigenvector of BA with the same
  eigenvalue:
  BA(Bv) = B(ABv) = B(\lambda v) = \lambda (Bv)
  This only fials when Bv = 0, but in this case ABv = 0 so \lambda = 0.
  Thus we can concldue that any non-zero eigenvalues of AB are also eigenvalues of BA

  It's much computationally cheaper to compute the eigvenvalues of BA (since
  it's a much smaller matrix), and this gives us all the non-zero eigenvalues
  of AB

  Question (hard) - how can you easily compute the SVD of M?
  Hint
    For a size-(m,n) with m > n, the algorithmic complexity of finding
    SVD is O(mn^2). So it's relatively cheap to find the SVD of A and B
    (complexity mn^2 vs m^3). Can you use that the SVD of M?
  Answer
    It's much cheaper to compute the SVD of the small matrices A and B.
    Denote these SVDs by:
    A = U_A S_A V_A^T
    B = U_B S_B V_B^T
    where U_A and V_B are (m,n), and the other matrices are (n,n).
    Then we have:
    M = AB
      = U_A (S_A V_A^T U_B S_B) V_B^T
    Note that the matrix in the middle has size (n,n) (i.e. small), so we can
    compute its SVD cheaply:
    S_A V_A^T U_B S_B = U' S' V'^T
    and finally, this gives us the SVD of M:
    M = U_A U' S' V'^T V_B^T
      = U S V'^T
    where U = U_A U', V = V_B V', and S = S'

    All our SVD calculations and matrix multiplications had complexity at most O(mn^2),
    which is much better than O(m^3) (remember that we don't need to compute
    all the values of U = U_A U', only the ones which correspond to non-zero singular values).

  If you're curious, you can go to the FactoredMatrix documentation to see the implementation of the SVD
  calculation, as well as other properties and operations.

  Now that we've discussed some of the motivations behind having a `FactoredMatrix` class,
  let's see it in action.
"""

# %%

# Basic Examples
# We can use the basic class directly - 
# let's make a factored matrix directly and 
# look at the basic operations

if MAIN:
  A = t.randn(5, 2)
  B = t.randn(2, 5)
  AB = A @ B
  AB_factor = FactoredMatrix(A, B)
  print("Norms:")
  print(AB.norm())
  print(AB_factor.norm())

  print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")

# %%

# We can also look at the eigenvalues and singular
# values of the matrix. Note that, because the matrix
# is rank 2 but 5 by 5, the final 3 eigenvalues and 
# singular values are zero - the factored class omits the zeros

if MAIN:
  print("Eigenvalues:")
  print(t.linalg.eig(AB).eigenvalues)
  print(AB_factor.eigenvalues)

  print("\nSingular Values:")
  print(t.linalg.svd(AB).S)
  print(AB_factor.S)

  print("\nFull SVD:")
  print(AB_factor.svd())

# Aside - the sizes of objects returned by the SVD method
"""
If M = USV^T, and M.shape = (m, n) and the rank is r,
then the SVD method returns the matrices U, S, V.
They have shape (m, r), (r,), and (n, r) respectively,
"""

# %%

# We can multiply a factored matrix with an unfactored
# matrix (as in example below). We can also
# multiply two factored matrices together to get
# another factored matrix

if MAIN:
  C = t.randn(5, 300)
  ABC = AB @ C
  ABC_factor = AB_factor @ C

  print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
  print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
  print(f"\nRight dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")

# %%

# If we want to collapse this back to an unfactored matrix,
# we can use the AB proeprty to get the product:

AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)

# %%

# Reverse-engineering circuits

"""
With our induction circuit, we have four individual
circuits: the OV and QK circuits in our previous
token head, and the OV and QK circuits in our
induction head. In the following sections of the
exercise, we'll reverse-engineer each of these
circuits in turn.
- In the section OV copying circuit, we'll look
  at the layer-1 OV circuit.
- In the section QK prev-token circuit, we'll look
  at the layer-0 QK circuit
# The third section (K-composition) is a bit
  trickier, because it involves looking at the
  composition of the layer-0 OV circuit and layer-1 QK circuit.
  We will have to do two things:
  1. Show that these two circuits are composing
     (i.e. that the output of the layer-0 OV circuit
     is the main determinant of the key vectors in the
     layer-1 Qk circuit)
  2. Show that the joint operation of these two
     circuits is "make the second instance of
     a token attend to the token following an
     earlier instance"
  The dropdown below contains a diagram explaining
  how the three sections relate to the different
  components of the induction circuit. You might
  have to open it in a new tab to see it clearly.

  Diagram
  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described_2_new.png
"""

# [1] OV copying circuit

"""
Let's start with an easy parts of the circuit - the copying OV circuit of 1.4 and 1.10. Let's start
with head 4. The only interpretable (read: privileged basis)
things here are the input tokens and output logits,
so we want to study the matrix:
W_EW_{OV}^{1.4}W_U
(And same for 1.10). This is the (d_vocab, d_vocab)-shape
matrix that combines with the attention pattern to get us
from input to output.
We want to calculate this matrix, and inspect it.
We should find that its diagonal values are very high,
and its non-diagonal values are much lower.

Question - why should we expect this observation?
you may find it helpful to refer back to the previous
section, where you described what the interpretation
of different matrices was.
Hint
  Suppose our repeating sequences is A B ... A B.
  Let A, B the corresponding one-hot encoded tokens.
  The B-th row of this matrix is:
  B^TW_EW_{OV}^{1.4}W_U
  What is the interpretation of this expression,
  in the context of our attention head?
Answer
  If our repeating sequence is A B ... A B, then:
  B^TW_EW_{OV}^{1.4}W_U
  is the vector of logits which gets moved from
  the first B token to the second A token, to be used
  as the prediction for the token following the second A token.
  It should result in a high prediction for B, and a low
  prediction for everything else. In other words,
  the (B, X)-th element of this matrix should
  be highest for X=B, which is exactly what we claimed.

  If this still seems confusing, the diagram below
  might help:
  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described-OV-v3.png
"""

# %%

# Exercise - compute OV circuit for 1.4

"""
This is the first of several similar exercises
where you calculate a circuit by multiplying
matrices. This exercise is pretty important (in particular,
you should make sure you understand what this matrix
represents and why we're interested in it), but the
actual calculation sohuldn't take very long.

You should compute it as a FactoredMatrix object.

Remember, you can access the model's weights directly
e.g. using model.W_E or model.W_Q (the latter
gives you all the W_Q matrices, indexed by layer and head)

Help - I'm not sure how to use this class to compute a product of more than 2 matrices.

  ```
  full_OV_circuit = FactoredMatrix(W_E @ W_V, W_O @ W_U)
  ```

  Alternatively, another nice feature about the FactoredMatrix class is that you can chain
  together matrix multiplicaiton. The following code defines
  exactly the same FactoredMatrix object:

  ```
  OV_circuit = FactoredMatrix(W_V, W_O)
  full_OV_circuit = W_E @ OV_circuit @ W_U
```
"""

if MAIN:
  # Solution
  head_index = 4
  layer = 1

  W_E = model.W_E
  W_V = model.W_V[layer, head_index]
  W_O = model.W_O[layer, head_index]
  W_U = model.W_U

  OV_circuit = FactoredMatrix(W_V, W_O)
  full_OV_circuit = W_E @ OV_circuit @ W_U

  tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

# %%

"""
Now we want to check that this matrix is the identity.
Since it's in factored matrix form, this is a bit tricky,
but there are still things we can do.

First, to validate that it looks diagonal-ish, let's
pick 200 random rows and columns and visualise that -
it sohuld at least look identity-ish here! We're using
the indexing method of the FactoredMatrix class -
you can index into it before returning the actual
`.AB` value, to avoid having to compute the whole thing
(we take advantage of the fact that 
`A[left_indices, :] @ B[:, right_indices]` is the same
as `(A @ B)[left_indices, right_indices]`).
"""

if MAIN:
  indices = t.randint(0, model.cfg.d_vocab, (200,))
  full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

  imshow(
    full_OV_circuit_sample,
    labels={
      "x": "Logits on output token",
      "y": "Input token"
    },
    title="Full OV circuit for copying head",
    width=700,
    height=600,
  )

"""
Personal note
  A bright spot at position (row, column) = (50, 50) means:
  When the input token is token #50, this circuit
  produces a large positive logit for output token
  #50."
  The fact that this is true for all points
  along the diagonal means the circuit consistently
  copies any input to the output

Aside - indexing factored matrices

  Yet another nice thing about factored matrices is 
  that you can evaluate small submatrices without having
  to compute the entire matrix. This is based on
  the fact that the [i, j]th element of matrix AB is
  A[i, :] @ B[:, j]
"""

# %%

"""
Exercise - compute circuit accuracy

When you index a factored matrix, you get back another factored matrix.
So rather than explicitly calculating A[left_indices, :] @ B[:, left_indices],
we can just write AB[left_index, left_indices]

you should observe a pretty distinct diagonal pattern here,
which is a good sign. However, the matrix is pretty
noise so it probably won't be exactly the identity.
Instead, we should come up with a summary statistic
to capture a rough sense of "closeness to the identity".

Accuracy is a good summary statistic - what fraction
of the time is the largest logit on the diagonal?
Even if there's lots of noise, you'd probably still expect
the largest logit to be on the diagonal a good deal of the time

If you're on a Colab or have a powerful GPU, you should
be able to compute the full matrix and perform this test.
However, it's beter practice to iterate through this matrix
when we can, so that we avoid CUDA issues. We've
given you a batch_size argument in the function below,
and you sohuld try to explicitly calculate matrices of size batch_size * d_vocab
rather than the massive matrix of d_vocab * d_vocab

Help - I'm not sure whether to take the argmax over rows or columns
  The OV circuit is defined as W_E @ W_OV @ W_U.
  We can see the i-th row W_E[i] @ W_OV @ W_U as the
  vector representing the logit vector added at any
  token which attends to the `i`-th token, via the attenion head
  with OV matrix `W_OV`.

  So we want to take the argmax over rows (i.e. over `dim=1`),
  because we're interested in the number of tokens `tok`
  in the vocabulary such that when `tok` is attended to,
  it is also the top prediction

This should return about 30.79% - pretty underwhelming. It goes up to 47.73% for top-5, but still not great.
What's up with that?
"""

def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
  """
  Return the fraction of the time that the maximum value is on the circuit diagonal
  """
  total = 0

  for indices in t.split(t.arange(full_OV_circuit.shape[0], device=device), batch_size):
    AB_slice = full_OV_circuit[indices].AB
    total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()

  return total / full_OV_circuit.shape[0]

if MAIN:
  print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")

# %%

"""
Exercise - compute effective circuit

Now we return to why we have two induction heads. If both have
the same attention pattern, the effective OV circuit
is actually W_E(W_V^{1.4}W_O^{1.4} + W_V^{1.10}W_O^{1.10})W_U,
and this is what matters. So let's re-run our analysis on this!

https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/effective_ov_circuit.png

Question - why might the model want to split the cirrcuit across two heads?
  Because W_VW_O is a rank 64 matrix. The sum of two is a rank 128 matrix.
  This can be a signfiicantly better approximation to
  the desired 50K x 50K matrix.

Expected output
  You should get an accuracy of 95.6 for top-1 - much better!

  Note that you can also try top 5 accuracy, which improves your result to 98%.
"""

if MAIN:
  W_O_both = einops.rearrange(
    model.W_O[1, [4, 10]],
    "head d_head d_model -> (head d_head) d_model"
  )
  W_V_both = einops.rearrange(
    model.W_V[1, [4, 10]],
    "head d_model d_head -> d_model (head d_head)"
  )

  W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

  print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(W_OV_eff):.4f}")

# %%

"""
[2] QK prev-token circuit

The code below plots the full QK circuit for head `0.7`
(including a scaling and softmax step, 
which is meant to mirror how the QK bilinear form
will be used in actual attention layers).
You should run the code and interpret the results
in the context of the induction circuit.
"""

if MAIN:
  layer = 0
  head_index = 7

  # Compute full QK matrix (for positional embeddings)
  W_pos = model.W_pos
  W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
  pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

  # Mask, scale and softmax the ccores

  mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
  pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

  # Plot the reuslts
  print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
  imshow(
    utils.to_numpy(pos_by_pos_pattern[:200, :200]),
    labels={
      "x": "Key",
      "y": "Query",
    },
    title="Attention patterns for prev-token QK circuit, first 200 indices",
    width=700,
    height=600,
  )

"""
The expected output and interpretation

The ful lQK circuit W_posW_{QK}^{0.7}W_pos^T has shape
[n_ctx, n_ctx]. It is a bilinear form, with the
(i, j)-th element representing the attention score
paid by the i-th token to the j-th token. This should
be very large when j = i - 1 (and smaller
for all other values of j), because this is a
previous head token. So if we softmax over j,
we should get a lower-diagonal stripe of 1.

https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described-QK-v4.png

Why is it justified to ignore token encodings?
In this case, it turns out that the positional
encodings have a much larger effect on the
attention scores than the token encodings.
If you want, you can verify this for yourself -
after going through the next section (reverse-engineering
K-composition), you'll have a better
sense of how to perform attribution on the inputs
to attendion heads, and assess their importance
"""

# %%

"""
[3] K-composition circuit

We now dig into the hard part of the circuit -
demonstrating the K-Composition between the
previous token head and the induction head.

Splitting activations
We can repeat the trick from the logit attribution scores.
The QK-input for layer 1 is the sum of 14 terms
(2+n_heads) - the token embedding, the positional
embedding, and the results of each layer0 head.
So for each head H in layer 1, the query tensor
(ditto key) corresponding to sequence position i is:
xW_Q^{1.H} = (e + pe + \sum_{h=0}^{11} x^{0.h})W_Q^{1.H}
           = eW_Q^{1.H} + peW_Q^{1.H} + \sum_{h=0}^{11} x^{0.h}W_Q^{1.H}

where e stands for the token embedding, pe
for the positional embedding, and x^{0.h} for the output
of head h in layer 0 (and the sum of these tensors
equals the residual stream x). All these tensors
have shape [seq, d_model]. So we can treat the
expression above as a sum of matrix multiplications
[seq, d_model] @ [d_model, d_head] -> [seq, d_head]

For ease of notation, I'll refer to the 14 inputs as
(y_0, y_1, ..., y_{13}) rather than (e, pe, x^{0.h}, ..., x^{h.11}). So we have:
xW_Q^h = \sum_{i=0}^{13} y_iW_Q^h

with each y_i having shape [seq, d_model]

https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/components.png
"""

# %%

"""
Exercise - analyse the relative importance

We can now analyse the relative importance of
these 14 terms! A very crude measure is to take the
norm of each term (by component and position).

Note that this is a pretty dodgy metric - q and k
are not inherently interpretable! But it can be
a good and easy-to-compute proxy

Question - why are Q and K not inherently interpretable?
Why might the norm be a good metric in spite of this?
  They are not inherently interpretable because they
  operate on the residual stream, which doesn't have
  a privileged basis. You could stick a rotation
  matrix R after all the Q, K and V weights (and
  stick a rotation matrix before everything that
  writes to the residual stream), and the model
  would still behave exactly the same

  The reason taking the norm is still a reasonable
  thing to do is that, despite the individual elements
  of these vectors not being inherently interpretable,
  it's still a safe bet that if they are larger than
  they will have a greater overall effect on the
  residual stream. So loooking at the norm doesn't
  tell us how they work, but it does indicate which
  ones are more important

What you should see
  YOu should see that the most important query
  components are the token and positional embeddings.
  The most important key components are those
  from y_9, which is x_7, i.e. from head 0.7

A technical note on the positional embeddings - optional,
feel free to skip this.
  You might be wondering why the tests compare the decomposed
  qk sum with the sum of the resid_pre + pos_embed,
  rather than just resid_pre. The answer lies in
  how we defined the transformer, specifically in this line
  from the config:
  ```
  positional_embedding_type="shortformer"
  ```
  The result of this is that the positional embeddings
  isn't added to the residual stream. Instead, it's added
  as inputs to the Q and K calculation (i.e. we
  calculate `(resid_pre + pos_embed) @ W_Q`
  and same for `W_K`), but not as inputs to the V
  calculation (i.e. we just calculate resid_pre @ W_V).
  This isn't actually how attention works in general,
  but for our purposes it makes the analysis
  of induction heads cleaner because we don't have
  positional embeddings interfering with the OV circuit.
"""

def decompose_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
  """
  Retrieves all the input tensors to the first attention layer, and concatenates them along the 0th dim.

  The [i, :, :]th element is y_i (from notation above). The sum of these tensors along the 0th dim should
  be the input to the first attention layer.
  """
  y0 = cache["embed"].unsqueeze(0) # shape (1, seq, d_model)
  y1 = cache["pos_embed"].unsqueeze(0) # shape (1, seq, d_model)
  y_rest = cache["result", 0].transpose(0, 1) # shape (12, seq, d_model)

  return t.concat([y0, y1, y_rest], dim=0)


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values).
    """
    W_Q = model.W_Q[1, ind_head_index]
  
    return einops.einsum(
      decomposed_qk_input,
      W_Q,
      "n seq d_model, d_model d_head -> n seq d_head"
    )


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    """
    W_K = model.W_K[1, ind_head_index]

    return einops.einsum(
      decomposed_qk_input,
      W_K,
      "n seq d_model, d_model d_head -> n seq d_head",
    )

if MAIN:
  # Recompute rep tokens/logits/cache, if we haven't already
  seq_len = 50
  batch_size = 1
  (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
  rep_cache.remove_batch_dim()

  ind_head_index = 4

  # First we get decomposed q and k input, and check they're what we expect
  decomposed_qk_input = decompose_qk_input(rep_cache)
  decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
  decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)
  t.testing.assert_close(
    decomposed_qk_input.sum(0), 
    rep_cache["resid_pre", 1] + rep_cache["pos_embed"],
    rtol=0.01,
    atol=1e-05,
  )
  t.testing.assert_close(
    decomposed_q.sum(0),
    rep_cache["q", 1][:, ind_head_index], 
    rtol=0.01, 
    atol=0.001
  )
  t.testing.assert_close(
    decomposed_k.sum(0), 
    rep_cache["k", 1][:, ind_head_index], 
    rtol=0.01, 
    atol=0.01
  )

  # Second, we plot our results
  component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
  for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
      utils.to_numpy(decomposed_input.pow(2).sum([-1])), # squared L2 norm (or squared Euclidean length)
      labels={"x": "Position", "y": "Component"},
      title=f"Norms of components of {name}",
      y=component_labels,
      width=800,
      height=400,
    )

"""
This tells us which heads are probably important,
but we can do better than that. Rather than
looking at the query and key components separately,
we can see how they combine together - i.e. take
the decomposed attention scores.

This is a bilinear function of q and k, and so we
will end up with a decompsoed_scores tensor with shape
(query_component, key_component, query_pos, key_pos),
where summing along BOTH of the first axes 
will give us the original attention scores (pre-mask)
"""

# %%

"""
Exercise - decompose attention scores

Implement the function giving the decomposed
scores (remember to scale by sqrt(d_head)!) For now, don't mask it.

Question - why  do I focus on the attention scores,
not the attention pattern? (i.e. pre softmax not post softmax)

  Because the decomposition trick only works
  for things that are linear - softmax isn't linear
  and so we can no longer consider each component
  independently.

Help - I'm confused about what we're doing / why we're doing it
  Remember that each of our components writes to the
  residual stream separately. So after layer 1, we have:
  
  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/components.png

  We're particularly interested in the attention scores
  computed in head `1.4`, and how they depend on
  the inputs into that head. We've already
  decomposed the residual stream value x into
  its terms e, pe, and x_0 through x_11 (which
  we've labelled y_0, ..., y_13 for simplicity),
  and we've done the same for key and query terms.
  We can picture these terms being passed into
  head `1.4` as:

  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/components-2.png

  So when we expand `attn_scores` out in full,
  they are a sum of 14^2 = 196 terms - one
  for each combination of (query_component, key_component).

  Why is this decomposition useful?
  We have a theory about a particular circuit in our model.
  We think that head `1.4` is an induction head,
  and the most important components that feed
  into this head are the prev token head `0.7` (as key)
  and the token embedding (as query). This is already
  supported by the evidence of our magnitude plots
  above (because we saw that `0.7` as key
  and token embeddings aquery were lerge), but
  we still don't know how this particular key and query
  work together; we've only looked at them separately

  By decomposing `attn_scores` like this, we can check whether
  the contribution from combination `(query=tok_emb, key=0.7)`
  is indeed producing the characteristic induction head
  pattern which we've observed (and the other 195 terms don't really matter)
"""

def decompose_attn_scores(
    decomposed_q: Float[Tensor, "q_comp q_pos d_head"],
    decomposed_k: Float[Tensor, "k_comp k_pos d_head"],
    model: HookedTransformer,
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, 0, 0]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    """
    return einops.einsum(
      decomposed_q,
      decomposed_k,
      "q_comp q_pos d_head, k_comp k_pos d_head -> q_comp k_comp q_pos k_pos"
    ) / (model.cfg.d_head**0.5)

if MAIN:
  tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)

# %%

if MAIN:
  # Once these tests have passed, you can plot the results:

  # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7), you can replace this
  # with any other pair and see that the values are generally much smaller, i.e. this pair dominates the attention score
  # calculation
  decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k, model)

  q_label = "Embed"
  k_label = "0.7"
  decomposed_scores_from_pair = decomposed_scores[component_labels.index(q_label), component_labels.index(k_label)]

  imshow(
      utils.to_numpy(t.tril(decomposed_scores_from_pair)),
      title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
      width=700,
  )

  # Second plot: std dev over query and key positions, shown by component. This shows us that the other pairs of
  # (query_component, key_component) are much less important, without us having to look at each one individually like we
  # did in the first plot!
  decomposed_stds = einops.reduce(
      decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
  )
  imshow(
      utils.to_numpy(decomposed_stds),
      labels={"x": "Key Component", "y": "Query Component"},
      title="Std dev of attn score contributions across sequence positions<br>(by query & key component)",
      x=component_labels,
      y=component_labels,
      width=700,
  )

"""
Personal note
  Why do we need to reduce it with `t.std`?
    We need a way to measure the "importance" or
    "activity level" of each of the 196 component
    pairs. We want to know which pairs
    are actually doing something and which are just
    producing noise or near-zero scores.
  Why not `mean`?
    The average (mean) attention score could be close
    to zero even if the scores themselves are large but
    balanced between positive and negative. A mean of
    zero doesn't necessarily mean "inactive"

Help - I don't understand the interpretation of these plots.
  The first plot tells you that the term 
  eW_{QK}^{1.4}(x^{0.7})^T
  (i.e. the component of the attention scores for head
  `1.4` where the query is supplied by the token embeddings
  and the key is supplied by the output of head `0.7`)
  produces the distinctive attention pattern we see
  in the induction head: a strong
  diagonal stripe.

  Although this tells us that this component would
  probably be sufficient to implement the
  induction mechanism, it doesn't tell us the whole
  story.
  Ideally, we'd like to show that the other
  195 terms are unimportant. Taking the standard
  deviation across the attention scores for a particular
  pair of ocmponents is a decent proxy for how
  important this term is in the overall attention
  pattern. The second plot shows us that the standard
  deviation is very small for all the other components,
  so we can be confident that the other components
  are unimportant.

  To summarise:
  - The first plot tells us that the pair
    (q_component=tok_emb, k_component=0.7)
    produces the characteristic induction head pattern
    we see in attention head `1.4`
  - The second plot confirms that this pair is the only
    important one for influencing the attention pattenr
    in `1.4`; all other pairs have very
    small contributions
"""

# %%

"""
Note that plots like the ones above are often the
most concise way of presenting a summary of the
important information, and understanding what
to plot is a valuable skill in any model internals-based
work. However, if you want to see the "full plot"
which the two plots above are both simplifications
of in some sense, you can run the code below which
gives you the matrix of every single pair of components'
contribution to the attention scores. So the first
plot above is just a slice of the full plot below,
and the second plot above is just below after
reducing over each slice with the standard deviation
operation

(Note - the plot you'll generate below is pretty big,
so you'll want to clear it after you're done with it.
If your machine is still workign slowly when
rendering it, you can use
fig.show(config={"staticPlot": True})
to display a non-interactive version of it
"""

if MAIN:
  decomposed_scores_centered = t.tril(decomposed_scores - decomposed_scores.mean(dim=-1, keepdim=True))

  decomposed_scores_reshaped = einops.rearrange(
    decomposed_scores_centered,
    "q_comp k_comp q_token k_token -> (q_comp q_token) (k_comp k_token)",
  )

  fig = imshow(
      decomposed_scores_reshaped,
      title="Attention score contributions from all pairs of (key, query) components",
      width=1200,
      height=1200,
      return_fig=True,
  )
  full_seq_len = seq_len * 2 + 1
  for i in range(0, full_seq_len * len(component_labels), full_seq_len):
      fig.add_hline(y=i, line_color="black", line_width=1)
      fig.add_vline(x=i, line_color="black", line_width=1)

  fig.show(config={"staticPlot": True})

"""
Personal note
  Why are we doing decomposed_scores_centered?
    The goal is to remove biases and improve
    the visual contrast of the final plot.

    Imagine a specific query component (e.g.,
    Query=PosEmbed) that, for some reason,
    tends to produce a large positive score for all
    key positions. It has a high "baseline interest".
    In the raw plot, this entire row of `101 * 14 = 1414`
    pixels would be bright, washing out any subtle,
    more important patterns with it.

    By centering the scores, we are asking a more refined
    question:
    - Before Centering: "What is the raw attention
      score from query `A` to key `B`?"
    - After Centering: "How much more or less
      than its own average does query `A` attend to
      key `B`?"
    
    This normalization makes the truly significant
    interactions stand out. A bright spot on the
    centered plot now represents a query-key interaction
    that is exceptionally strong relative to that
    query's baseline behavior. It highlights the specific,
    non-uniform patterns (like the induction stripe)
    while suppressing the uniform, less informative ones.
    It's a standard technique in data visualization
    to enhance contrast and reveal underlying structure.
"""

# %%

"""
Interpreting the ufll circuit
Now we know that head `1.4` is composing with
head `0.7` via K composition, we can
multiply through to create a full circuit:

W_EW_{QK}^{1.4}(W_{OV}^{0.7})^TW_E^T
and verify that it's the identity.
(Note, when we say identity here, we're again
thinking about it as a distribution over logits, so
this should be taken to mean "high diagonal values",
and we'll be using our previous metric of `top_1_acc`.)

Question - why should this be the identity?
  This matrix is a bilinear form. Its diagonal elements (A, A) are:
  A^TW_EW_{QK}^{1.4}W_{OV}^{0.7}W_E^TA = (A^TW_EW_{Q}^{1.4})(A^TW_EW_{OV}^{0.7}W_K^{1.4})^T
  
  Intuitively, the query is saying "I'm looking for a token which followed A",
  and the key is saying "I am a token which followed A" (recall that
  A^TW_EW_{OV}^{0.7} is the vector which gets moved one position
  forward by our prev token head `0.7`).

  Now, consider the off-diagonal elements (A, X) (for X \neq A).
  We expect these to be small, because the key doesn't match the query:

  A^TW_EW_{QK}^{1.4}W_{OV}^{0.7}W_E^TX = (I'm looking for a token which followed A) \cdot (I am a token which followed X)

  Hence, we expect this to be the identity.
  An illustration:
  https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/kcomp_diagram_described-K-last.png
"""

# %%

"""
Exercise - compute the K-comp circuit

Calculate the matrix above, as a `FactoredMatrix` object

Aside about multiplying FactoredMatrix objects together
  If `M1 = A1 @ B1` and `M2 = A2 @ B2` are factored matrices,
  then `M = M1 @ M2` returns a new factored matrix.
  This might be:
  ```
  FactoredMatrix(M1.AB @ M2.A, M2.B)
  ```
  or it might be:
  FactoredMatrix(M1.A, M1.B @ M2.AB)
  with these two objects corresponding to the
  factorisations M = (A_1B_1A_2)(B_2) and M = (A_1)(B_1A_2B_2) respectively

  Which one gets returned depends on the size of the hidden
  dimension, e.g. M1.mdim < M2.mdim then the factorisation
  used will be M = A_1B_1(A_2B_2).

  Remember that both these factorisations are valid,
  and will give you the exact same SVD. The only
  reason to prefer one over the other is for
  computational efficiency (we prefer a smaller
  bottleneck dimension, because this determines the
  computational complexity of operations like
  finding SVD)
"""

def find_K_comp_full_circuit(
    model: HookedTransformer, 
    prev_token_head_index: int, 
    ind_head_index: int,
) -> FactoredMatrix:
    """
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side (direct from token
    embeddings) and the second dimension being the key side (going via the previous token head).
    """
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]

    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K

    return FactoredMatrix(Q, K.T)

prev_token_head_index = 7
ind_head_index = 4
K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")

"""
You can also try this out for our other induction head
`ind_head_index=10`, which should also return
a relatively high result. Is it higher than for head `1.4`?

Note - unlike last time, it doesn't make sense
to consider the "effective circuit" formed by adding
together the weight matrices for heads `1.4` and
`1.10`. Can you see why?
  Because the weight matrices we're dealing with here
  are from the QK circuit, not the OV circuit.
  These don't get combined in a linear way;
  instead we take softmax over each head's
  QK circuit output individually
"""

# %%
