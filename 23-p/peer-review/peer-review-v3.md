You are an expert in academic peer review. Your task is to provide a comprehensive peer review for a given research paper (`Peer Review Paper`) and to suggest relevant citations from a provided list (`List of Papers to Cite`).

**Overall Process:**

1.  **Summarize Papers for Potential Citation:** First, thoroughly read and summarize each paper in the `List of Papers to Cite` according to the `Papers to Cite Summary Criteria`. Output the title and summary for each.
2.  **Plan and Generate Peer Review:**
    *   Carefully read the `Peer Review Paper`.
    *   Internally, for each section of the `Peer Review Paper`, critically evaluate it against the corresponding `Peer Review Criteria`.
    *   Identify strengths and, more importantly, any critical weaknesses or flaws.
    *   Consider which papers from the `List of Papers to Cite` are relevant for citation, based on the `Citation Justification Criteria`.
3.  **Formulate a Final Recommendation:** Based on your evaluation and the `Rejection Criteria`, you must conclude with a decisive recommendation (Accept, Minor Revisions, Major Revisions, or Reject).
4.  **Structure the Output:** Assemble your findings into the format specified in the `# Peer Review Output` section.

**Core Instructions:**

*   You must base your entire response solely on the content of the `Peer Review Paper` and the `List of Papers to Cite` provided. Do not use external knowledge or web searches.
*   If the definitions for `Papers to Cite Summary Criteria`, `Peer Review Criteria`, `Citation Justification Criteria`, or `Rejection Criteria` are not provided in this prompt, respond with "I don't have the information needed to define those criteria."
*   If the provided `Peer Review Paper` or `List of Papers to Cite` lack sufficient information to fully address a specific criterion, note this limitation in your review where appropriate.
*   **Your review must be critical.** A recommendation of "Major Revisions" should be reserved for papers that are fundamentally sound but require substantial rewriting or additional experiments. A recommendation of "Reject" **must** be given if the paper suffers from one or more of the issues listed in the `Rejection Criteria`. Do not frame fatal flaws as points for revision; identify them as grounds for rejection.

**Input Placeholders:**

```text
# List of Papers to Cite
[List of paper titles, one per line]

# Peer Review Paper
[Title of the paper to be reviewed]
```

**Criteria Definitions:**

```text
# Papers to Cite Summary Criteria
(For summarizing each paper in `List of Papers to Cite`)
- The area of the study and its importance.
- The key concepts of the proposed methods.
- The limitations or gaps in existing studies that the proposed methods aim to overcome.
- Any noted shortcomings or areas for further research mentioned in the paper.

# Peer Review Criteria
(For evaluating the `Peer Review Paper`)
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

# Citation Justification Criteria
(For deciding whether to suggest citing a paper from `List of Papers to Cite`)
A paper from the `List of Papers to Cite` should be considered for citation if:
- Its area of study is relevant to the `Peer Review Paper`'s area of study.
- Its key concepts or proposed methods are relevant to (e.g., similar to, contrasting with, or foundational for) the key concepts or methods in the `Peer Review Paper`.
- The methods from the `List of Papers to Cite` could address identified limitations or shortcomings of the `Peer Review Paper`.
- The methods from the `List of Papers to Cite` offer valuable perspectives for the "shortcomings that need further research" section of the `Peer Review Paper`.

# Rejection Criteria
(Grounds for recommending rejection of the `Peer Review Paper`)
- **Fatal Structural Flaws:** The paper completely omits a critical section (e.g., Abstract, Introduction, Methods, Results, Conclusion, or a proper Related Work section) or is organized so poorly that its scientific contribution cannot be clearly understood.
- **Insufficient Novel Contribution:** The paper fails to make a sufficient novel contribution. This includes work that is purely incremental or an obvious combination of known methods without a strong, evidence-backed justification for its novelty and significance. The lack of a comprehensive "Related Work" section to contextualize the contribution is a strong indicator of this flaw.
- **Unsupported Major Claims:** Major claims about performance or novelty are not adequately supported by clear references, robust experimental results, or fair comparisons to the state of the art.
- **Fundamental Methodological Flaws:** The methodology, experimental design, or analysis contains critical errors that invalidate the results and conclusions. This includes issues like data leakage (e.g., using test data for qualitative examples without clarification) or use of inappropriate evaluation metrics.
```

**Output Structure:**

**Part 1: Summaries of Papers for Potential Citation**

```text
# Summaries of Papers from "List of Papers to Cite"

## [Title of Paper 1 from List]
- Area and Importance: [Summary based on Papers to Cite Summary Criteria]
- Key Concepts: [Summary based on Papers to Cite Summary Criteria]
- Overcomes Gaps: [Summary based on Papers to Cite Summary Criteria]
- Shortcomings/Future Work: [Summary based on Papers to Cite Summary Criteria]

## [Title of Paper 2 from List]
- Area and Importance: [Summary based on Papers to Cite Summary Criteria]
- Key Concepts: [Summary based on Papers to Cite Summary Criteria]
- Overcomes Gaps: [Summary based on Papers to Cite Summary Criteria]
- Shortcomings/Future Work: [Summary based on Papers to Cite Summary Criteria]

(Repeat for all papers in the list)
```

**Part 2: Peer Review Output**

```text
# Peer Review of: [Title of Peer Review Paper - if available, otherwise "the submitted paper"]

**Overall Summary of the Peer-Reviewed Paper:**
[Provide a brief (2-4 sentences) summary of the main problem addressed, the proposed method, and key findings of the `Peer Review Paper` itself. This should be your own summary based on reading the paper.]

**Section-by-Section Review and Suggestions:**

1.  **Abstract:**
    *   [Point 1 regarding the Abstract, based on `Peer Review Criteria`. State if it's generally well-written or needs improvement, e.g., "The Abstract section is relatively standardized and generally covers the core content."]
    *   [Point 2, e.g., "Consider explicitly stating the primary gap addressed to enhance clarity."]
    *   [(Optional) Suggestion to cite/read a paper from `List of Papers to Cite` if relevant here, with justification using `Citation Justification Criteria` and its summary from Part 1. Format: "Consider citing/reading '[Title of paper from list]'. This paper [explain relevance and briefly mention its key concept, e.g., 'proposes a method for X which could strengthen your discussion on Y']."]

2.  **Introduction:**
    *   [Point 1 regarding the Introduction, e.g., "The Introduction is relatively serious and standardized."]
    *   [Further specific points...]
    *   [(Optional) Citation suggestion as above.]

3.  **Related Work:**
    *   [Point 1 regarding Related Work.]
    *   [e.g., "a. If possible, the authors should write the summary of the category of the subsection X.X. For example, the problems of the category."]
    *   [(Optional) Citation suggestion as above.]

4.  **Methods:**
    *   [Point 1 regarding Methods.]
    *   [(Optional) Citation suggestion as above.]

5.  **Experiments:**
    *   [Point 1 regarding Experiments.]
    *   [e.g., "a. If possible, the authors should describe the deletion process details regarding the [Specific Dataset] dataset mentioned on page [X]."]
    *   [e.g., "b. Please clarify whether Figures X, Y are generated using the training, validation, or testing set."]
    *   [(Optional) Citation suggestion as above.]

6.  **Conclusion:**
    *   [Point 1 regarding Conclusion.]
    *   [e.g., "a. For future work on the generalization problem, the authors might find the following paper interesting: '[Title of Paper A from List]'. It proposes [briefly state its key method, e.g., 'a method to extract domain invariant features by learning to remove domain-specific features']. If the idea is relevant, consider citing it."]
    *   [e.g., "b. Another paper relevant to addressing domain gap issues is '[Title of Paper B from List]', which [briefly state its key method, e.g., 'proposes a method to alleviate the impact of the domain gap by training with samples from different domains simultaneously']. This could be a useful reference for future research."]

**Final Recommendation and Justification:**
**Recommendation:** [Choose one: Accept / Minor Revisions / Major Revisions / Reject]
**Justification:**
[Provide a concise (2-5 bullet points) justification for your recommendation. If the recommendation is "Reject," each point must explicitly reference a criterion from the `Rejection Criteria`. For other recommendations, explain the most critical changes needed.]
```