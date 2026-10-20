# Limitations, Ethical Considerations, and Governance

## 1. Non-Production Research Scope
This system is an academic research deliverable developed as an internship capstone. It is strictly evaluated as a proof-of-concept and is not certified for automated, autonomous production decision-making.

## 2. Dataset Limitations & Temporal Shift
* **Dataset Age:** The underlying corpus (Amazon Fine Food Reviews) covers October 1999 through October 2012.
* **Lexical Drift:** Terminology, brand names, and dietary vocabulary introduced post-2012 are absent from the baseline corpus.
* **Rating Distribution Skew:** Historical ratings heavily favor positive reviews (78.0% positive, 14.5% negative, 7.5% neutral), causing inherent class imbalance.

## 3. Label Noise & Binarization Constraints
* **Binarization Artifacts:** Training excluded 3-star reviews to enforce binary separation (positive vs. negative). Ambiguous or mixed-polarity text is forced into binary classification.
* **Sarcasm and Incongruence:** Reviews exhibiting positive text with low star ratings (or vice versa) create label noise within supervised checkpoints.

## 4. Privacy, Bias & Safety Guardrails
* **Data Privacy:** Data was sourced exclusively from public historical archives.
* **Read-Only Agent Guardrails:** The AI Analyst interface enforces read-only operations across all tools, blocking destructive commands, file modifications, and prompt injections (32/32 safety test cases passed).