# Note

1. `main.ipynb` is not trained on the laptop. The model was trained on a computer with NVIDIA RTX 3090.
2. The computer took 34.5 minutes just to train for 5000 iterations + output 300 tokens.
3. The computer took 8 minutes just to output 10,000 tokens. See `more.txt` to check the output.
4. Mind you, the `input.txt` is only 1 million characters (and because we use character as token, then it's 1 million tokens) / 300k tokens (by tokenizer standard used by GPT-3 which have 50,000 vocabulary and trained on 300 billions tokens). See the "Language Models are Few-Shot Learners" paper.

# Reference:

1. [Transformer](https://www.youtube.com/watch?v=kCc8FmEb1nY)
2. [BatchNorm](https://youtu.be/DtEq44FTPM4?feature=shared&t=249)

> If without BatchNorm, suppose we have height and age as axes. The mean and std of height is not 0 and higher than 1, respectively. This property also applies to axes. Then, if we plot it, the circle will be shifted and elongated. This makes training difficult (because the optimal area is very small and not in the center). So, why we use LayerNorm instead of BatchNorm in Transformer?

1. BatchNorm vs. LayerNorm:
- BatchNorm normalizes the activations across the batch dimension (i.e., along the samples in a mini-batch). The mean and standard deviation are calculated over the entire batch for each feature.
  - This works well for Convolutional Neural Networks (CNNs) where the mini-batch contains many images with similar characteristics (e.g., pixel distributions).
- LayerNorm, on the other hand, normalizes the activations across the feature dimension for each individual sample. That is, for each input, the mean and standard deviation are computed across the features (e.g., the embedding dimensions in a Transformer model), rather than across the mini-batch. This means every sample is normalized independently.

2. Why BatchNorm Doesn’t Work Well for Transformers:
Now, let's break down why LayerNorm is preferred in Transformers over BatchNorm:

A. Sequence Length Variability:
- In Transformers, the sequence length (T, the number of tokens in a sentence or a sequence) can vary greatly depending on the input.
- BatchNorm normalizes the activations across the mini-batch, so for each feature (e.g., each embedding dimension), it computes the mean and standard deviation over all the sequences in the batch. This means BatchNorm is dependent on the batch size and the variance of the sequences in the batch.
- Since sequence lengths can vary, the statistics computed by BatchNorm could be misleading or inconsistent, which can harm performance. LayerNorm, on the other hand, computes statistics independently for each token or sequence, so it doesn't have this dependency on the batch size or sequence length.

B. No Spatial Structure in Transformers:
- BatchNorm is great when there is a spatial structure to the data, as in images, where the activations for different pixels in an image (across the mini-batch) are related to each other in a structured way.
- In Transformers, however, the tokens in a sequence are not spatially related in the same way as pixels in an image. Each token is treated as a separate unit, and there is no inherent spatial relationship between them. As a result, BatchNorm could lead to improper normalization of individual tokens because it mixes information across sequences.

C. Training Dynamics:
- BatchNorm computes statistics over the batch, which means that the normalization depends on the batch size. Small batch sizes lead to noisy statistics, making training more unstable. On the other hand, LayerNorm normalizes over individual tokens, so it is not as sensitive to batch size.

- In Transformers, since the batch size is often relatively small in practice (e.g., due to memory limitations), LayerNorm becomes more stable than BatchNorm. BatchNorm’s reliance on batch statistics can be problematic when training with small batches.

D. No Shift or Elongation of the Distribution:
- As you correctly noted, in the case where the data is not centered and has a large variance (e.g., height and age with non-zero means and standard deviations greater than 1), applying normalization across the batch would result in a shifted and elongated distribution of the activations. This could make the optimization process difficult because the effective area of the loss landscape becomes small and shifted away from the optimal region.
- LayerNorm normalizes the activations within each sample across the feature dimensions, which helps avoid this shift and elongation problem. This way, each token in the sequence has its own normalization, ensuring that the optimization process is more stable and efficient.

3. Summary of Why LayerNorm is Used in Transformers:
- LayerNorm is applied across the feature dimension for each token independently, which is crucial for sequence-based tasks like in Transformers.
BatchNorm struggles with variable sequence lengths, batch size sensitivity, and the lack of a spatial structure in sequential data like text.
- LayerNorm provides more stable training because it’s independent of the batch size and does not require batch statistics, making it ideal for models like Transformers, which process sequences with varying lengths and are not bound by spatial correlations between elements.

Conclusion:

While BatchNorm can work well for convolutional layers and image data (where spatial relationships exist), LayerNorm is better suited for sequence-based models like Transformers. It normalizes the data at the level of each token, not across the batch, which provides the model with stable and effective training regardless of batch size or sequence length variability. This is why LayerNorm is preferred in the Transformer architecture.

> so, what is the rationale behind using the LayerNorm in Transformer? what is the use of it?

3. Improvement of Gradient Flow
In deep networks, especially those with many layers like the Transformer, it’s common to face issues such as vanishing or exploding gradients, which can slow down training or make it unstable. LayerNorm improves gradient flow by stabilizing the learning process:

- By normalizing the input to each layer (across features), it ensures that the data entering the next layer is well-conditioned (i.e., has a mean of 0 and a standard deviation of 1). This reduces the risk of gradients exploding or vanishing during backpropagation.
- This is especially important in deep architectures like Transformers, where there are multiple layers of attention and feed-forward networks. LayerNorm ensures that the learning process doesn’t become dominated by certain features that have disproportionately large magnitudes.

6. Helping with Identity Mapping and Residual Connections
Transformers utilize residual connections in both the multi-head attention and the feed-forward layers. The purpose of residual connections is to allow the model to easily learn the identity mapping (i.e., to pass information directly without transformation if needed), which helps with training deep networks.

- LayerNorm is used after the residual connection (i.e., the output of the layer is added to its input and then normalized). This helps the model easily decide whether the residual path should be skipped or modified.
- Since the residual connection can help the model retain important information from earlier layers, LayerNorm ensures that the activations passing through these residual connections are normalized and stay well-conditioned.

7. Improving Non-linearity and Expressiveness
- The ReLU activation function introduces non-linearity into the model, and LayerNorm helps maintain a stable and consistent representation after each layer of transformation.
- By normalizing activations, LayerNorm helps each layer produce consistent outputs, which allows the model to learn non-linear transformations effectively without being hindered by changes in the scale or variance of the activations. This is important in deep architectures like the Transformer, where maintaining stability in activations and gradients is crucial for effective learning.

8. Generalization and Regularization
- LayerNorm acts as a form of regularization. By reducing the dependence of each token's representation on the distribution of the others in the batch, it introduces a certain level of noise into the training process, which helps prevent overfitting.
- It effectively ensures that each token is treated independently, encouraging the model to learn generalizable patterns rather than memorizing batch-specific statistics.

Summary:
The rationale behind using LayerNorm in Transformers is:

- Stability: It improves the stability of training by normalizing the activations and gradients within each sequence.
- Independence from batch size and sequence length: LayerNorm doesn’t depend on batch statistics, making it ideal for sequence-based data with variable lengths.
- Improved learning: It helps maintain stable gradient flow, reduces vanishing/exploding gradients, and accelerates convergence in deep networks.
- Effective for non-spatial data: Unlike BatchNorm, which is designed for spatially structured data (like images), LayerNorm is better suited for sequence data like text, where each token in the sequence is independent.

In short, LayerNorm in Transformers improves the training dynamics, ensures stability, and enhances the model's ability to generalize by normalizing across the feature dimensions of each token independently. This is essential for handling the varying lengths and complexities of sequence data, making it an ideal choice for architectures like Transformers.

> In the original Transformer architecture from the "Attention is all you need" paper, the LayerNorm is applied after the Multi-Head Attention and also after the Feed Forward. but in some models, the LayerNorm is applied before the Multi-head Attention and also before the Feed Forward. why is that? what is the intention behind it? what is the rational behind it? what is the use of it? what is the difference between before and after?

In the original Transformer architecture, as proposed in the "Attention is All You Need" paper, LayerNorm is applied after the Multi-Head Attention and Feed-Forward sub-layers, which is part of a design known as the Post-LayerNorm approach. However, some modified models use LayerNorm before the Multi-Head Attention and Feed-Forward sub-layers (called the Pre-LayerNorm approach).

Let's dive into the differences between the two, the rationale behind each approach, and the effects on training.

1. Post-LayerNorm (Original Transformer Approach)
In the original Transformer architecture, LayerNorm is applied after the operations of both the Multi-Head Attention and Feed-Forward layers:

- Residual Connection: First, the input to the Multi-Head Attention or Feed-Forward layer is added to the output (residual connection).
- Layer Normalization: Then, LayerNorm is applied on the result of the residual addition.

Why is LayerNorm applied after the operations in the original Transformer?

- Gradient Flow: In the Post-LayerNorm design, the residual connection allows the model to learn identity mappings more easily. This means that if the network doesn’t need a transformation in some cases, it can simply learn to pass the input as is through the residual connection. LayerNorm after the residual ensures that the output activations are normalized before they are passed to the next layer, maintaining stable training dynamics.
- Stable activations: By normalizing the activations at the end of the sub-layer, the model benefits from consistent and well-conditioned representations, which improves the convergence during training and helps with stability.
- Independence from the activation dynamics: The order ensures that the residual connection is not disturbed by the normalization process and helps in retaining useful information that might otherwise be lost in deep networks.

Summary of Post-LayerNorm:
- LayerNorm is applied after the main computation (Multi-Head Attention and Feed-Forward).
- It stabilizes the output activations.
- Helps gradients flow more easily and maintains the integrity of the residual connection.

2. Pre-LayerNorm (Modified Models)

In some variants of Transformer models (e.g., Reformer, GPT-2, and others), LayerNorm is applied before the Multi-Head Attention and Feed-Forward layers:

- LayerNorm is applied before the operations of the sub-layers (Attention and Feed-Forward).
- Residual Connection: Then, the result of the layer is added to the input via the residual connection, and that becomes the final output for that sub-layer.

Why LayerNorm before the sub-layers (Pre-LayerNorm)?
The Pre-LayerNorm approach alters the way the normalization interacts with the input data:

- Normalization of inputs: By normalizing the inputs before the sub-layers (Attention or Feed-Forward), this approach ensures that the activations entering each sub-layer are already normalized. This can help prevent issues related to exploding/vanishing gradients early in training and could provide more stable learning when training deep networks.
- More consistent scale: Since the normalization is done before the sub-layer operation, the input features will always have zero mean and unit variance. This can prevent issues that arise from learning large weights early in the training, especially in deep networks where the weights could get too large or small if not normalized properly.
- Stabilization of activations: In deeper models, Pre-LayerNorm can help reduce instability by ensuring that activations entering each sub-layer are already centered around zero with unit variance, making them easier to process in subsequent layers.
- Easier to train very deep models: Pre-LayerNorm helps improve the optimization process in very deep networks by ensuring that the inputs to each layer are well-conditioned (i.e., normalized) before the computation starts. This makes training more efficient and helps avoid issues that might arise from unnormalized or large activations.
- Easier to train very deep models: Pre-LayerNorm helps improve the optimization process in very deep networks by ensuring that the inputs to each layer are well-conditioned (i.e., normalized) before the computation starts. This makes training more efficient and helps avoid issues that might arise from unnormalized or large activations.

Summary of Pre-LayerNorm:
- LayerNorm is applied before the multi-head attention and feed-forward operations.
- Normalizes the input activations to each sub-layer, helping with training stability.
- Can help train deeper models more effectively by ensuring each layer’s input is well-conditioned.

3. Key Differences Between Post-LayerNorm and Pre-LayerNorm:


Aspect	| Post-LayerNorm	| Pre-LayerNorm
--- | --- | ---
Where LayerNorm is applied	| After the residual connection (output of attention/feed-forward).	| Before the residual connection (input to attention/feed-forward).
Effect on residual connections	| The residual connection is kept intact, meaning the original input can directly contribute to the output. |	The input is normalized first, and then the residual connection is added afterward.
Training stability	| Stabilizes the outputs by normalizing after the computation, helping gradients flow more easily.	| Stabilizes the inputs to the computations by normalizing them first, which can improve the efficiency of training, especially for deeper models.
Gradient flow	| Gradients can flow more smoothly due to post-layer normalization and direct contribution of residuals.	| Helps to ensure that gradients are well-conditioned from the start by normalizing the input features.
Common Use	| Used in the original Transformer model.	| Used in some recent Transformer variants (e.g., GPT-2, Reformer).

4. Why Choose Pre-LayerNorm or Post-LayerNorm?
- Pre-LayerNorm is often preferred in deeper Transformer models because it provides greater stability during training and improves the optimization process by ensuring the input to each layer is normalized and well-conditioned. This makes it easier to train large models efficiently.
- Post-LayerNorm is the approach used in the original Transformer, where the residual connection is preserved, and normalization happens after the sub-layer operations. It is known to work well in practice and has been foundational in Transformer-based models.

5. Conclusion:
- Post-LayerNorm: LayerNorm is applied after the computation in each sub-layer, ensuring stable and well-conditioned outputs that benefit from the residual connection. This approach is used in the original Transformer architecture.

- Pre-LayerNorm: LayerNorm is applied before the computation, normalizing the inputs to each sub-layer to stabilize training. This approach is used in certain newer models and helps with training deeper networks more effectively.

The choice between Pre-LayerNorm and Post-LayerNorm depends on the model's architecture and the specific challenges (like training depth, stability, and optimization efficiency) that need to be addressed during training. Both have their advantages, but Pre-LayerNorm is gaining popularity, particularly for deeper models, because it stabilizes the learning process from the very beginning of each layer.

> just give me the exact reason, do not write a novel. give me the core concept why we need a layer norm in a transformer architecture? don't use ambiguity word like "well conditioned activations". remember, what is the core concept why we need a layer norm in a transformer architecture? think carefully

The core concept behind using LayerNorm in a Transformer architecture is to normalize the activations in order to prevent the gradient from becoming too large or too small during training, ensuring stable learning. It standardizes the outputs (or inputs, depending on pre or post) by making sure they have zero mean and unit variance, which helps the model train effectively by maintaining consistent scale across layers. This avoids issues like exploding/vanishing gradients and accelerates convergence.