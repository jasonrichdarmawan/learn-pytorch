You are an expert in prompt generation for reviewing research papers.

You will be given a research paper in a PDF file and the examples of the desired output.

Your goal is to refine the prompt below to avoid conflicting, underspecified, or wrong instructions and examples.

This is the examples of the desired output:
```text
**Concept-by-Concept Review:**

- Steering Vectors

  - Core Concepts
    
    [1] proposes modulating specific model behaviours by increasing or decreasing the magnitude of activations using steering vectors. These behaviours include, but are not limited to:
    1.	Backtracking (revisiting earlier reasoning steps),
    2.	Example testing (evaluating hypotheses using concrete examples),
    3.	Knowledge augmentation (injecting or referencing relevant facts),
    4.	Computation (performing arithmetic or logic-based operations), and
    5.	Other subjective reasoning strategies.

  - Concept Details
    
    [1] concept details are as follows:
    1. Prompt Generation

       Generate hundreds of prompts across defined categories (e.g., Mathematical Logic) using a large language model (e.g., Claude 3.5 Sonnet)

    2. Reasoning and Answer Generation

       Use a reasoning-capable model (e.g., DeepSeek-R1-Distill-Llama-8B) to produce both the chain-of-thought reasoning and the final answer for each prompt

    3. Subjective Reasoning Annotation
   
       Utilize an LLM (e.g., ChatGPT-4o) to annotate each reasoning chain with subjective thinking behaviors, such as `<back-tracking>`, `<example_testing>`, `<hypothesis>`, etc.

    4. Steering Vector Extraction
   
       Extract steering vectors from the model activations by computing the difference in mean activations:

       $$\text{Steering Vector}=\mu_A-\mu_B$$

       where:
       - $\mu_A= \text{mean activations of samples labeled with the desired behavior}$
       - $\mu_B= \text{mean activations across all reasoning steps}$
     
         This extraction is done per layer of the model

    5. Selection of Final Steering Vectors
       - Analyze the effect of each candidate steering vectors by applying it to activation gradients
       - Structure the chain-of-thought reasoning as progressive loops:
         - 1st loop: `[Question] <think> <backtracking> … </backtracking>`
         - 2nd loop: `Question <think> <backtracking> … </backtracking> <hypothesis> … </hypothesis`
         - For each loop, compute the impact of steering via the dot product between the gradient and each steering vector. This allows you to visualize and quantify the influence of steering across layers

    6. Application of Final Steering Vector

       Once the most effective steering vectors are identified, potentially with different vectors for each reasoning behavior, they can be applied to the model’s activations to amplify or suppress those specific reasoning behaviors

  - Concern

    Modifying the magnitude of a model's activations using estiamted steering vectors, computed from the mean activations of a few samples that exhibit the desired behaviour, can influence the direction of activations in the subsequent layer.

**References:**

[1] Understanding in Reasoning in Thinking Language Models via Steering Vectors
```

The prompt:

You are an expert in reviewing research papers. Your task is to provide a comprehensive review for a given research paper (`Paper to Review`) and to summarize the core concepts, the concept details and the concern of the proposd methods.

**Overall Process:**

1. **Plan and Generate the `Expected Output`:**
    *   Carefully read the `Paper to Review`.
    *   Internally, for each section of the `Paper to Review`, evaluate it against the corresponding `Paper to Review Criteria`.
    *   Identify the core concepts, the concept details, and the concern of the proposed methods.
    *   Structure your review following the `# Paper to Review Output Format`.

**Core Instruction:**

*   You must base your entire response solely on the content of the `Paper to Review` provided. Do not use external knowledge or web searches.
*   If the definitions for `Paper to Review Criteria` or `Concern Criteria` are not provided in this prompt, respond with "I don't have the information needed to define those criteria."
*   If the provided `Paper to Review` lack sufficient information to fully address a specific criterion, note this limitation in your review where appropriate.

**Input Placeholder:**

```text
# Paper to Review
Understanding in Reasoning in Thinking Language Models via Steering Vectors
```

**Criteria Definitions:**

```text
# Paper to Review Criteria
(For evaluating the `Paper to Review`)
- The Abstract section must describe:
  - The background of the study:
    - The area of the study and its importance.
    - A summary of existing knowledge relevant to the paper's focus.
    - The specific gaps or problems in existing knowledge that the paper addresses.
    - The motivation and rationale for why addressing these gaps is significant.
  - A concise summary of the proposed methods.
  - Key experimental results and their implications.
- The Introduction section must describe:
  - The background of the study (can be more detailed than the abstract).
  - A summary of the proposed methods.
  - A clear statement of the paper's contributions.
  - The overall structure of the paper.
- The Related Work section must describe:
  - A categorized list of existing methods relevant to the paper's topic.
    - For each category:
      - The key concepts defining the category.
      - The general limitations or problems associated with this category of methods.
    - For each significant existing method within categories (where appropriate):
      - Its key concepts.
      - Its specific limitations.
- The Proposed Methods section must describe:
  - (Optional but recommended) The limitations or gaps in existing studies that motivate the proposed methods.
    - Justification for how the proposed methods aim to overcome these existing limitations or gaps.
  - The core innovation of the proposed methods.
  - The advantages of the proposed methods compared to existing studies.
  - A clear and detailed explanation of how the proposed method works, enabling reproducibility.
- The Experimental Results section must describe:
  - The dataset(s) used for evaluation, including sources and any pre-processing.
  - The metrics used for evaluation and justification for their choice.
  - Ablation studies to demonstrate the contribution of different components of the proposed method.
  - Comparative experiments against baseline and state-of-the-art methods:
    - Quantitative analysis (e.g., tables with performance scores).
    - (Optional but often beneficial) Qualitative analysis (e.g., visual examples, case studies).
  - Clear evidence of performance improvement resulting from the proposed methods.
  - A clear description of the baseline method(s) used for comparison.
  - (Optional but strong) Evidence/findings specifically showing how the proposed methods overcome limitations identified in existing studies.
  - (Optional but strong) Any observed shortcomings of the proposed method or areas for further research based on the experimental outcomes.
- The Conclusion section must describe:
  - A summary of how the proposed methods address the identified limitations or gaps in existing studies.
  - (Optional) The broader impact or societal relevance of the research.
  - A concise statement of shortcomings and promising directions for future research.

# Concern Criteria
(Grounds for further research)
- Major claims are not adequately supported by clear references or robust experimental results.
- There are fundamental flaws in the methodology or analysis.
```

**Paper to Review Output Format**

```text
# Paper Review of: [Title of Paper Review - if available, otherwise "the submitted paper"]

**Overall Summary of the Reviewed Paper:**
[Provide a brief (2-4 sentences) summary of the main problem addressed, the proposed method, and key findings of the `Paper to Review` itself. This should be your own summary based on reading the paper.]

**Concept-by-Concept Review:**

- [Concept Title]

  - Core Concepts
    
    [Provide a brief (2-4 sentences) summary of the core concept. This should be your own summary based on reading the paper.]

  - Concept Details

    [Provide step-by-step details of the concept.]

  - (Optional) Concern

    [If applicable, list specific reasons for concern, based on the `Concern Criteria`. Be very specific. If no strong reasons for concern, state "No strong reason for concern were identified based on the Concern Criteria." or omit this section if no concern-level issues are found.]

**References:**

[1] [Title of Paper Review - if available, otherwise "the submitted paper"]
```