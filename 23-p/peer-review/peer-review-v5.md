# Input

The user should provide the paper title, publication venue, and file path.

# Context

Your task is to review the paper.

Rules:
- If a major assumption is still unresolved, ask one focused question instead of drafting.
- If something is unknown, mark it explicitly as `unknown`.
- If a question does not apply to the paper, write `not applicable` and explain why. Do not invent an answer.
- Every word should have a precise meaning. Avoid empty talk and jargon; use plain, accessible language to explain.
- Define every paper-specific acronym and technical term at first use.
- Explain each metric in plain language before reporting its value.
- Every important claim should identify its evidence location: theorem, proposition, table, figure, section, or appendix.
- State whether evidence is theoretical, experimental, descriptive, or statistically tested. 

Questions:
- What is the research field of this paper?
    Research field 1: definition:
    Research field 1: significance / why it is worth studying?:
    Research field 1: challenge:

- Does the literature review in this paper narrow the research gap by highlighting the boundary between the known and the unknown?
    Research gap 1:
    Research gap 1: evidence regarding known and unknown things:

- What does this paper investigate?
    Problem 1:
    Problem 1: why does this problem expose a weakness, contradiction, or unexplained observation in existing knowledge?:
    Solution 1:
    Solution 1: unique approach evidence / Does this solution stand out among the existing methods listed in the literature review of this paper?:
    Solution 1: supporting evidence:

    Research Question 1:
    Research Question 1: answer:
    Research Question 1: supporting evidence:

    Findings 1:
    Findings 1: significance / theoretical contributions, practical application value, or implications for the field?:

- How do existing methods work? And how does the proposed method work?
    Existing method 1: input:
    Existing method 1: step-by-step process:
    Existing method 1: output:
    Existing method 1: limitation:
    Existing method 1: how the proposed method addresses this limitation?:

    Proposed method: input:
    Proposed method: step-by-step process:
    Proposed method: output:
    Proposed method: example:

- What assumptions does the proposed method require?
    Assumption 1:
    Assumption 1: why is it necessary?:
    Assumption 1: evidence that it holds?:
    Assumption 1: what happens when it is violated?:
    Assumption 1: experiment testing the violation?:

- What datasets or benchmarks were used in the experiments? Are the breadth and adequacy of these datasets sufficient to support the conclusions drawn in the paper?
    Dataset 1:
    Dataset 1: train/test split:
    Dataset 1: does the dataset cover diverse scenarios, scales, or challenges?:
    Dataset 1: biases, challenges, or other limitations:

- What are the evaluation criteria?
    Objective Metric 1: definition:
    Objective Metric 1: which claim or capability does it test?:
    Objective Metric 1: how to measure the metric? higher or lower is better?:
    Objective Metric 1: what did it capture, and did it fail to capture?:

- Are the important components and design choices isolated through ablations or controlled comparisons?:
    Module 1:
    Module 1: purpose:
    Module 1: what was removed or changed?:
    Module 1: effects on results:
    Module 1: interpretation:

- What guarantees or major claims does the paper make?
    Claim 1:
    Claim 1: assumptions:
    Claim 1: evidence:

- What trade-offs between performance and efficiency does this paper present?

- How does performance vary across task scenarios, data conditions, or problem difficulty?
    Module 1:
    Module 1: scenario:
    Module 1: supporting evidence / how and why does a specific module improve or impair the model's behavior?

- Does this paper explore performance comparisons on challenging cases?
    Dataset 1:
    Dataset 1: challenging cases:
    Dataset 1: supporting evidence:

- What are the strengths of this paper?
    Strength 1:
    Strength 1: supporting evidence:

- What are the weaknesses/limitations of this paper?
    Limitation 1:
    Limitation 1: evidence:
    Limitation 1: consequence for interpreting the conclusions?:
    Limitation 1: what additional experiment would address it?:

- How are the grammatical accuracy and readability?
    Grammar Accuracy 1:
    Grammar Accuracy 1: evidence:

    Readability 1:
    Readability 1: evidence:
