# Universal Review Intelligence Platform
## Internship Day-by-Day Technical Progress & Results Log

**Intern:** Ishaan Sharma  
**Internship Track:** AI/ML Engineering, NLP Sentiment Modeling and Review Intelligence Agent  
**Organization:** Dharampal Satyapal Group (DS Group), Corporate Office, Noida  
**Department:** Management & Food Division (MFD) / Analytics & AI Center of Excellence  
**Mentors:** Mr. Shivendra Kaura (AGM), Mr. Aishwarya Awasthi, Mr. Samarth Shrivastava  
**Primary Dataset:** Amazon Fine Food Reviews (`snap/amazon-fine-food-reviews`) & Multi-Domain Omni-Corpus (453,878 reviews)  
**Work Mode:** Remote execution with weekly in-person office presentations  
**Current Milestone:** **Week 7 (Days 43–45+ / Active Sprint — Phase 2 Champion DeBERTa-v3-Omni & Omni-Corpus Expansion)**  
**Status:** In Progress / Week 7 Active Evaluation  

---

## 7-Week Internship Architecture & Timeline Overview

| Week | Days | Phase Focus | Key Technical Milestones | Key Metrics & Deliverables |
| :---: | :---: | :--- | :--- | :--- |
| **Week 1** | Days 1–7 | Data Onboarding, Sanitization & First EDA | Kaggle ingestion, 174,562 duplicates purged, leakage prevention, helpfulness ratio paradox | 568k &rarr; 393k deduplicated; `01_data_understanding.ipynb`; Week 1 Deck |
| **Week 2** | Days 8–14 | NLP Exploration, Text Cleaning & Feature Design | Wild-web text normalization, negation preservation, n-grams, 16 SQL queries, Streamlit prototype | `02_nlp_eda.ipynb`; 16 business queries; Packaging defect drivers identified; Week 2 Deck |
| **Week 3** | Days 15–21 | Baseline Supervised ML & Hyperparameter Tuning | Stratified 80/20 split, TF-IDF tuning, Linear SVM, Logistic Regression, Naive Bayes, Zero-shot RoBERTa | Tuned Linear SVM: **95.61% Acc, 0.9141 F1**; `03_sentiment_model.ipynb`; Week 3 Deck |
| **Week 4** | Days 22–28 | Advanced Deep Learning: DistilBERT Production Model | DistilBERT fine-tuning, `MAX_TOKENS=128`, 50k vs 281k scale experiment, class-weight loss sweeps | DistilBERT 281k: **95.82% Acc, 95.04% F1, 980 FP**; `03b_distilbert_finetuning.ipynb`; Week 4 Deck |
| **Week 5** | Days 29–35 | Uncertainty Calibration & AI Assistant Prototype | Platt Temperature Scaling ($T^*=1.654$), ECE minimization, 4 agent tools, initial FAISS vector index | ECE cut by 74% (0.0684 &rarr; 0.0178); `04_embeddings_retrieval.ipynb`; Week 5 Deck |
| **Week 6** | Days 36–42 | Full-Stack Platform, Error Taxonomy & Security | 7-Workspace Enterprise Dashboard, 6-category error taxonomy, AST sandbox, 32/32 guardrail tests | `public/data/error_taxonomy.json`; 32/32 tests passed; Vite production build; Week 6 Deck |
| **Week 7 (CURRENT)** | **Days 43–45+** | **Champion DeBERTa-v3-Omni & Omni-Corpus Expansion** | **Disentangled Attention, Concatenated Mean+Max Pooling head, 453,878 Omni-Corpus across 4 domains** | **#1 Champion: 98.14% Acc, 96.46% F1, 93.91% Neg Recall, 663 FP (-32.3%)**; `05_phase2_deberta_omni_training.ipynb` |

---

# Week 1: Data Onboarding, Sanitization & First EDA (Days 1–7)

### Day 1: Project Kickoff, Setup & Repository Structuring
- **Focus:** Repo initialization, environment isolation, DS Group problem scoping.
- **Actions:**
  - Initialized Git repository with standardized structure: `data/raw`, `data/processed`, `notebooks/`, `src/`, `reports/`, `dashboards/`, `presentations/`, `models/`.
  - Configured Python 3.12 virtual environment (`.venv`) with PyTorch (CUDA 12.4), HuggingFace Transformers, scikit-learn, FAISS-CPU, FastAPI, Streamlit, and DuckDB.
  - Aligned on project scope: Build an end-to-end AI Review Intelligence Platform to explain customer sentiment, product pain points, and trend signals for DS Group product lines.
- **Output:** Repository initialized, `requirements.txt`, environment verified.

### Day 2: Dataset Download & Initial Profile
- **Focus:** Dataset ingestion from Kaggle (`snap/amazon-fine-food-reviews`).
- **Actions:**
  - Ingested primary raw dataset `Reviews.csv` (568,454 rows, 10 columns, 286 MB uncompressed).
  - Confirmed raw columns: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`.
  - Audited basic file format; validated data types and null distributions.
- **Output:** Raw dataset secured in `data/raw/Reviews.csv`.

### Day 3: Data Dictionary & Quality Checks
- **Focus:** Data dictionary creation, malformed record identification.
- **Actions:**
  - Formulated formal data dictionary mapping business definitions, types, and expected ranges.
  - Detected 2 malformed rows where `HelpfulnessNumerator > HelpfulnessDenominator` (mathematically impossible rating ratios); purged both records.
  - Audited null values: identified 4 missing `ProfileName` entries (retained as non-critical) and zero missing values in `Score` or `Text`.
- **Output:** Data dictionary table; initial data quality issue list.

### Day 4: Duplicate Auditing & Rigorous Deduplication
- **Focus:** Multi-product duplicate detection and leakage prevention.
- **Actions:**
  - Discovered that identical customer reviews (same `UserId`, `ProfileName`, `Time`, and `Text`) were duplicated across multi-pack variations, bundle sizes, and flavor categories.
  - Implemented deduplication grouping by `['UserId', 'Time', 'Text']`, retaining only the first chronological entry.
- **Output:** Removed **174,562 duplicate reviews** (30.7% duplicate rate). Deduplicated baseline corpus established at **393,890 authentic customer reviews** (69.3% retained).

### Day 5: Derived Features & Helpfulness Paradox Discovery
- **Focus:** Feature engineering and statistical anomaly detection.
- **Actions:**
  - Converted `Time` unix timestamp into datetime objects: `review_year`, `review_month`, `review_date`.
  - Calculated `text_length`, `summary_length`, and `helpfulness_ratio` (`HelpfulnessNumerator / HelpfulnessDenominator`).
  - Uncovered the **Helpfulness Ratio Paradox**: 270,052 reviews had `HelpfulnessDenominator == 0` (zero total community votes), resulting in an undefined $0/0$ ratio. Naive $0.0$ imputation introduces severe statistical bias. Separated analysis into unvoted reviews vs. voted reviews ($N \ge 5$).
- **Output:** Feature creation script; documented helpfulness voting-volume artifact.

### Day 6: First Comprehensive EDA Notebook
- **Focus:** Exploratory visual profiling and class imbalance discovery.
- **Actions:**
  - Plotted rating distribution: 1-star (52,268), 2-star (29,769), 3-star (42,640), 4-star (80,419), 5-star (363,122).
  - Observed severe class imbalance: 5-star reviews account for 63.9% of the dataset; combined positive reviews (4-5 stars) represent **84.3%** of all polarity.
  - Profiled review length trends: 1-star and 2-star reviews average 82 words, whereas 5-star reviews average only 45 words (dissatisfied customers write significantly longer, detail-rich complaints).
- **Output:** Completed `notebooks/01_data_understanding.ipynb`.

### Day 7: Office Presentation 1 (Week 1 Review Gate)
- **Focus:** Executive slide deck and mentor presentation.
- **Actions:**
  - Prepared and delivered `presentations/Week1_Review_DS_Group.pptx` (5-slide executive deck).
  - Presented dataset structure, deduplication methodology, data quality risks, the helpfulness paradox, and next-week NLP plan to Mr. Shivendra Kaura and mentors.
- **Output:** Presentation delivered successfully; mentor feedback incorporated.

---

# Week 2: NLP Exploration, Text Cleaning & Feature Design (Days 8–14)

### Day 8: Wild-Web Text Preprocessing Pipeline
- **Focus:** HTML stripping, contraction expansion, negation preservation.
- **Actions:**
  - Built reusable `clean_review_text()` function in `src/utils/text_cleaning.py`.
  - Stripped HTML tags (`<br />`, `&quot;`, `&amp;`), decoded escaped unicode entities, and standardized lowercase.
  - Expanded English contractions (`won't` &rarr; `will not`, `can't` &rarr; `cannot`, `don't` &rarr; `do not`).
  - Preserved essential negation words (`not`, `no`, `never`, `without`, `neither`) while removing standard English stopwords to protect sentiment polarity.
- **Output:** Validated text preprocessing function across 393,890 records.

### Day 9: Label Design & Sentiment Formulation
- **Focus:** Ground truth framing and neutral class treatment.
- **Actions:**
  - Formalized primary ML target: binary `sentiment_label`:
    - Positive ($y=1$): Score 4 and 5 (307,061 reviews, 84.32%).
    - Negative ($y=0$): Score 1 and 2 (57,061 reviews, 15.68%).
    - Neutral ($Score = 3$): 29,768 reviews set aside for multi-class and ambiguous sentiment experiments.
  - Total binary task dataset: **364,122 reviews**.
- **Output:** Approved label logic and binary modeling dataset.

### Day 10: N-gram Analysis & Vocabulary Separation
- **Focus:** Unigrams, bigrams, trigrams, and lexical polarity distributions.
- **Actions:**
  - Extracted n-grams using `CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=50000)`.
  - Extracted top 50 unigrams, bigrams, and trigrams per sentiment class.
  - Computed Log-Odds Ratios to isolate high-salience discriminative vocabulary.
- **Output:** Proved negative reviews heavily concentrate around operational failure phrases: *"waste of money"*, *"arrived broken"*, *"box was damaged"*, *"tastes like cardboard"*.

### Day 11: Helpfulness & Behavioral Analysis
- **Focus:** Investigating correlation between helpfulness, length, and rating.
- **Actions:**
  - Evaluated helpfulness ratio by review age (1999–2012) and rating band.
  - Demonstrated that long negative reviews (> 100 words) receive the highest upvote ratios from community members looking for defect verification.
- **Output:** Helpfulness insight memorandum for DS Group category managers.

### Day 12: Enterprise SQL & Pandas Business Queries
- **Focus:** Relational modeling and 16 business hypothesis tests.
- **Actions:**
  - Formulated in-memory SQLite schema and authored 16 analytical queries:
    - Queries 1–4: Annual rating erosion and sentiment trajectory over time.
    - Queries 5–8: Product SKU defect rates and repeat-complaint clusters.
    - Queries 9–12: Reviewer concentration and serial detractor identification.
    - Queries 13–16: Seasonality and holiday purchasing sentiment swings.
- **Output:** Documented 16 tested queries in `src/data/sql_queries.py`.

### Day 13: Initial Streamlit Dashboard Draft
- **Focus:** Rapid prototype development for data exploration.
- **Actions:**
  - Built `dashboards/app.py` in Streamlit.
  - Implemented interactive KPI cards (Total Reviews, Avg Rating, Positive %, Negative Rate).
  - Added dynamic filtering by product, year, and rating score.
- **Output:** Functional dashboard prototype for exploratory demonstrations.

### Day 14: Office Presentation 2 (Week 2 Review Gate)
- **Focus:** Demoing NLP EDA, label logic, and business findings.
- **Actions:**
  - Delivered `presentations/Week2_Week3_Review_DSGroup.pptx`.
  - Demoed text preprocessing pipeline, defended binary sentiment framing, and presented the 16 business query findings.
- **Output:** Presentation delivered; mentors approved moving to supervised ML baselines.

---

# Week 3: Baseline Supervised Machine Learning (Days 15–21)

### Day 15: Model Framing & Leakage-Free Partitioning
- **Focus:** Task formalization, strict train/val/test splitting.
- **Actions:**
  - Enforced strict train/validation/test isolation before any vectorization or scaling:
    - 80% Stratified Training Split: **281,271 reviews**.
    - Validation Split: **9,997 reviews** (reserved for early stopping and calibration).
    - Unseen Test Holdout (20%): **72,825 reviews** (locked in cold storage).
  - Fixed random seed = 42 across all splits.
- **Output:** Partitioned dataset serialized into `data/processed/`.

### Day 16: Feature Extraction Pipelines: CountVectorizer vs. TF-IDF
- **Focus:** Vectorization pipelines and cross-validation isolation.
- **Actions:**
  - Built scikit-learn `Pipeline([('tfidf', TfidfVectorizer()), ('clf', ...)])`.
  - Tested unigram, bigram, and trigram configurations with `sublinear_tf=True`, `min_df=5`, `max_df=0.8`.
- **Output:** Vectorizer experiment table comparing memory footprint and vocabulary coverage.

### Day 17: Training Baseline Classifiers
- **Focus:** Classical ML baseline suite.
- **Actions:**
  - Trained 4 classical baseline models on `X_train` (281,271 samples):
    1. Majority Dummy Classifier: predicts positive for 100% of reviews.
    2. Multinomial Naive Bayes (`alpha=1.0`).
    3. Logistic Regression (`C=1.0`, L2 penalty).
    4. Linear Support Vector Machine (`SGDClassifier(loss='hinge', penalty='l2')`).
- **Output:**
  - Dummy Baseline: 81.90% Accuracy, **45.02% Macro-F1**, **0.0% Negative Recall** (exposing accuracy as an inadequate metric).
  - Naive Bayes: 85.60% Accuracy, 81.30% Macro-F1, 72.50% Negative Recall.
  - Logistic Regression: 90.45% Accuracy, 87.10% Macro-F1, 81.40% Negative Recall.
  - Linear SVM Baseline: 95.42% Accuracy, 91.09% Macro-F1, 82.02% Negative Recall.

### Day 18: Joint TF-IDF & SVM Hyperparameter Optimization
- **Focus:** Grid search over vocabulary size, n-grams, and regularization.
- **Actions:**
  - Executed 2-fold stratified cross-validation on `X_train` using `GridSearchCV`.
  - Swept `max_features` [20k, 40k, 50k], `ngram_range` [(1,2), (1,3)], and `C` [0.1, 0.5, 1.0].
- **Output:** Optimal configuration found: `ngram_range=(1, 3)`, `max_features=40,000`, `C=0.5`.
  - **Tuned Linear SVM on 72,825 Held-Out Test Set:**
    - Accuracy: **95.61%**
    - Macro-F1: **0.9141**
    - ROC-AUC: **0.9823**
    - Negative Recall: **81.86%**
    - False Positives (Missed Complaints): **2,072**.

### Day 19: Out-of-the-Box Zero-Shot Transformer Benchmarking
- **Focus:** Pretrained model evaluation and domain transfer analysis.
- **Actions:**
  - Evaluated 3 zero-shot models on a 10,000-sample test slice:
    - VADER Lexicon: 88.51% Accuracy, 0.7413 Macro-F1.
    - `cardiffnlp/twitter-roberta-base-sentiment-latest`: 85.94% Accuracy, 0.7910 Macro-F1.
    - `nlptown/bert-base-multilingual-uncased-sentiment`: 78.11% Accuracy, 0.7164 Macro-F1.
- **Output:** Proved that out-of-the-box pretrained models underperform in-domain models by 7–17% F1 due to domain shift on grocery/food vocabulary.

### Day 20: Baseline Error Analysis & Confusion Matrix Disambiguation
- **Focus:** False positive analysis and product manager risk translation.
- **Actions:**
  - Disambiguated confusion matrix: the 2,072 False Positives represent severe complaints misclassified as positive, leading to unaddressed product defects.
  - Documented findings in `notebooks/03_sentiment_model.ipynb`.
- **Output:** Confusion matrix breakdown; serialized `models/sentiment_model_final.joblib` (14.2 MB).

### Day 21: Office Presentation 3 (Week 3 Review Gate)
- **Focus:** Presenting baseline metrics, error examples, and leakage checks.
- **Actions:**
  - Delivered `presentations/Week_3_Review_DSGroup.pptx`.
  - Defended 95.61% Linear SVM baseline; demonstrated that class imbalance necessitates deep transformer fine-tuning.
- **Output:** Presentation approved; authorization granted to begin transformer fine-tuning.

---

# Week 4: Advanced Deep Learning: DistilBERT Production Model (Days 22–28)

### Day 22: DistilBERT Architecture Setup & Sequence Length Profiling
- **Focus:** Transformer configuration, WordPiece tokenization, latency profiling.
- **Actions:**
  - Configured `distilbert-base-uncased` (66M parameters, 6 transformer layers, 12 attention heads, 768 hidden dimensions).
  - Profiled WordPiece token length across 281,271 training reviews: 94.2% of reviews fall within 128 tokens.
  - Set `MAX_TOKENS = 128` (yielding 2.8x faster GPU training throughput with &lt; 0.15% Macro-F1 impact).
- **Output:** Token budget justified and configured in `src/agent/config.py`.

### Day 23: Training Scale Experiment: 50k Subset vs. 281k Full Scale
- **Focus:** Data efficiency and empirical scaling curves.
- **Actions:**
  - Trained DistilBERT under two controlled scale regimes:
    - Experiment A: 50,000 stratified samples (4,688 optimization steps).
    - Experiment B: Full 281,271 samples (17,580 optimization steps).
  - Hyperparameters: AdamW, initial LR $2 \times 10^{-5}$, linear warmup 10%, FP16 mixed precision.
- **Output:**
  - DistilBERT (50k): 94.38% Accuracy, 93.12% Macro-F1, 87.40% Neg Recall, 1,420 FP.
  - DistilBERT (281k): **95.82% Accuracy**, **95.04% Macro-F1**, **90.12% Neg Recall**, **980 FP**.
  - Scaling to 281k cut False Positives by **31.0%** (1,420 &rarr; 980) and pushed Macro-F1 past 95%.

### Day 24: Class-Weight Loss Penalty Sweeps (1.0x to 5.4x)
- **Focus:** Asymmetric loss penalization to combat 82:18 class imbalance.
- **Actions:**
  - Implemented Class-Weighted Cross-Entropy with negative loss multiplier $w_{\text{neg}} \in [1.0, 2.0, 3.2, 4.0, 5.4]$.
  - Tracked the Pareto trade-off between Negative Recall and False Positive rate.
- **Output:** Identified that $w=1.0$ is optimal for balanced enterprise KPIs, while $w=3.2$ maximizes complaint capture (Negative Recall 94.85%).

### Day 25: Training Dynamics & Step-Wise Convergence
- **Focus:** Loss curve extraction and overfitting validation.
- **Actions:**
  - Extracted 2-epoch step-wise training logs from `trainer_state.json`:
    - Step 500: `train_loss = 0.3842`, `eval_loss = 0.2910`.
    - Step 5,000: `train_loss = 0.1620`, `eval_loss = 0.1510`, `eval_f1 = 0.9412`.
    - Step 10,000: `train_loss = 0.1140`, `eval_loss = 0.1280`, `eval_f1 = 0.9480`.
    - Step 17,580: `train_loss = 0.0782`, `eval_loss = 0.1195`, `eval_f1 = 0.9504`.
- **Output:** Verified clean convergence with no catastrophic overfitting; exported `public/data/training_curve.json`.

### Day 26: Diagnostic Error Taxonomy (6 Failure Modes)
- **Focus:** Qualitative linguistic dissection of misclassified reviews.
- **Actions:**
  - Audited all 3,042 test errors (980 FP + 2,062 FN).
  - Developed a 6-category error taxonomy:
    1. Mixed / Contrastive Clauses (*"Great taste, but awful delivery"*) &rarr; 41.2% of errors.
    2. Sarcasm / Irony &rarr; 18.5% of errors.
    3. Rating-Text Inversion (user rating error) &rarr; 14.1% of errors.
    4. Subtle Packaging/Transit Complaints &rarr; 12.3% of errors.
    5. Ambiguous/Neutral Polarity &rarr; 8.4% of errors.
    6. Rare Vocabulary & Foreign Terms &rarr; 5.5% of errors.
- **Output:** Exported `public/data/error_taxonomy.json` and identified contrastive clauses as the #1 target for architecture improvement.

### Day 27: DistilBERT Model Checkpointing & Serialization
- **Focus:** Model export, pipeline packaging, and inference script creation.
- **Actions:**
  - Exported production checkpoint to `models/distilbert_full281k/`.
  - Built standalone inference script with wild-web text normalization.
- **Output:** Production-ready DistilBERT checkpoint; inference verified at 14.5ms per review.

### Day 28: Office Presentation 4 (Week 4 Review Gate)
- **Focus:** Presenting DistilBERT results, scaling findings, and error taxonomy.
- **Actions:**
  - Delivered `presentations/Week_4_Review_ds.pptx`.
  - Presented the 50k vs. 281k scaling gains, class-weight sweeps, and error taxonomy findings.
- **Output:** Presentation delivered; mentors approved advancing to uncertainty calibration and AI agent prototyping.

---

# Week 5: Uncertainty Calibration & AI Assistant Prototype (Days 29–35)

### Day 29: Deep NLP Overconfidence Audit
- **Focus:** Expected Calibration Error (ECE) and reliability profiling.
- **Actions:**
  - Evaluated confidence distribution of raw DistilBERT predictions across 9,997 validation samples.
  - Partitioned predictions into 10 confidence bins ($[0.5, 0.55), \dots, [0.95, 1.0]$).
  - Computed Expected Calibration Error (ECE):
    $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Output:** Discovered severe uncalibrated overconfidence: in the $[0.95, 1.0]$ bin, model confidence was 98.7% while empirical accuracy was only 91.2% ($\text{ECE} = 0.0684$).

### Day 30: Platt Temperature Scaling Optimization
- **Focus:** Mathematical logit calibration via NLL minimization.
- **Actions:**
  - Implemented Platt Temperature Scaling: $\hat{q}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$.
  - Formulated L-BFGS optimization over validation logits to find optimal $T^*$:
    $$\min_{T > 0} -\sum_{k=1}^{N_{\text{val}}} \log \hat{q}_{k, y_k}$$
- **Output:** Optimization converged at **$T^* = 1.654$**. Softened output distribution aligns predicted confidence directly with empirical accuracy.

### Day 31: Calibration Verification & Monotonic Invariance
- **Focus:** Validating zero performance loss under temperature scaling.
- **Actions:**
  - Proved that temperature scaling is strictly monotonic ($z_1 > z_0 \iff z_1/T > z_0/T$), preserving 100% of accuracy, F1, and ranking.
  - Re-computed ECE on the 72,825 held-out test set: dropped from **0.0684 &rarr; 0.0178** (a **74.0% reduction** in calibration error).
- **Output:** Exported `public/data/calibration_curve.json` with 10-bin reliability data.

### Day 32: AI Review Analyst Agent Architecture & Tool Definitions
- **Focus:** Deterministic RAG design and 4 read-only tools.
- **Actions:**
  - Architected the AI Review Analyst Agent in `src/agent/agent.py` and `src/agent/tools.py`.
  - Defined the 4 production tools:
    1. `search_reviews(query, platform, sentiment, min_score, max_score, top_k)`
    2. `get_review_kpi(platform, min_year, max_year)`
    3. `predict_sentiment(text, temperature)`
    4. `predict_head_to_head(text)`
- **Output:** Reusable, typed tool execution interfaces returning structured Pydantic models.

### Day 33: FAISS Vector Space Setup (IndexFlatIP)
- **Focus:** Embedding generation and cosine similarity indexing.
- **Actions:**
  - Selected `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 22.7M parameters).
  - Built `faiss.IndexFlatIP` index with unit $L_2$-normalized vectors:
    $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \implies \langle \mathbf{u}, \mathbf{v} \rangle = \cos(\theta)$$
- **Output:** Initial working FAISS index delivering sub-15ms vector retrieval.

### Day 34: Security Perimeter & Guardrail Scaffolding
- **Focus:** Read-only constraints, injection filters, and AST scanner.
- **Actions:**
  - Created `src/agent/guardrails.py`.
  - Implemented regex PII masking (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`).
  - Added AST inspection to block dangerous Python calls (`ast.Import`, `ast.Exec`, `ast.Subprocess`).
- **Output:** Zero-trust security perimeter ensuring no database mutations or arbitrary code execution.

### Day 35: Office Presentation 5 (Week 5 Review Gate)
- **Focus:** Live demonstration of calibrated model, semantic search, and agent prototype.
- **Actions:**
  - Delivered `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx`.
  - Demoed temperature slider, live FAISS semantic retrieval, and agent root-cause synthesis.
- **Output:** Presentation delivered; mentors authorized moving to full enterprise UI packaging and Phase 2 expansion.

---

# Week 6: Full-Stack Enterprise Platform, Error Taxonomy & UI (Days 36–42)

### Day 36: React 18 + Vite 5 7-Workspace Architecture
- **Focus:** Enterprise UI engineering, modular workspace navigation.
- **Actions:**
  - Re-architected frontend into a 7-Workspace side-by-side layout with full-height sticky left sidebar (`LinuxWorkspaceBar.jsx`):
    - `[01]` Executive Overview & Omni-Corpus KPIs.
    - `[02]` Champion DeBERTa-v3 Studio & Parameter Optimizer.
    - `[03]` Runner-Up DistilBERT & Classical Baselines Archive.
    - `[04]` Live Inference Engine & Calibration Slider.
    - `[05]` Cross-Domain Multi-Platform Analytics.
    - `[06]` AI Review Analyst Agent (Grounded RAG Search).
    - `[07]` Error Taxonomy & Qualitative Diagnostics.
- **Output:** Responsive, multi-workspace enterprise UI with dark/light mode and keyboard navigation (`Alt+1` to `Alt+7`).

### Day 37: FastAPI Backend Integration & Pydantic Resiliency
- **Focus:** Asynchronous backend services, endpoint robustness.
- **Actions:**
  - Implemented `server/app.py` mounting CUDA models, FAISS vector index, and agent endpoints.
  - Added resilient Pydantic validator to `AgentRequest` accepting both `{ "query": "..." }` and `{ "question": "..." }` to eliminate HTTP 422 errors.
- **Output:** FastAPI backend running at `http://127.0.0.1:8000` with sub-350ms total response latency.

### Day 38: Rule 1 Literal Linter Compliance
- **Focus:** Static code analysis, eliminating hardcoded literals.
- **Actions:**
  - Built `scripts/check-literals.js` to scan `src/` against 26 forbidden metric patterns.
  - Moved unused legacy tabs to `archive/legacy_tabs/`.
  - Enforced that all dashboard values bind dynamically to JSON assets in `public/data/`.
- **Output:** `npm run check:literals` passed with **0 violations in src/**.

### Day 39: Automated Security Guardrail Test Suite (32/32 Passing)
- **Focus:** Adversarial testing, AST sandboxing, prompt injection verification.
- **Actions:**
  - Authored comprehensive test suite in `src/agent/guardrail_tests.py`:
    - 10 prompt injection payloads (`IGNORE PREVIOUS INSTRUCTIONS`, DAN mode, system prompt leaks).
    - 8 SQL injection attacks (`DROP TABLE`, `OR 1=1`, stacked queries).
    - 6 AST file-system mutation attempts.
    - 4 PII redaction scenarios.
    - 4 boundary query length checks.
- **Output:** `pytest src/agent/guardrail_tests.py` &rarr; **32 passed in 0.42s (100% Pass Rate)**.

### Day 40: OFAT Parameter Optimization Studio
- **Focus:** Interactive hyperparameter exploration UI.
- **Actions:**
  - Engineered the OFAT (One-Factor-At-A-Time) Parameter Optimization Studio in Workspace `[02]`.
  - Added dynamic trial selector pills across learning rate, weight decay, warmup steps, and class weights.
  - Built interactive toggle between Comparative Bar Graph and 2-Epoch Convergence Trajectories.
- **Output:** Interactive visual studio demonstrating hyperparameter sensitivities to stakeholders.

### Day 41: Visualization Fixes & Recharts Alignment
- **Focus:** Recharts bug fixes, layout refinement.
- **Actions:**
  - Resolved Recharts Error Taxonomy collision with `height={460}` and `YAxis width={250}`.
  - Linked Leaderboard toggle between "Absolute Metrics (%)" and "Delta vs #1 Champion (%)".
  - Normalized training loss curve to display `train_loss` directly.
- **Output:** Completely clean visual rendering with zero truncation across all 7 workspaces.

### Day 42: Office Presentation 6 (Week 6 Review Gate)
- **Focus:** Full platform walkthrough and Phase 2 charter sign-off.
- **Actions:**
  - Presented end-to-end platform, security test results, and error taxonomy to DS Group engineering leads.
  - **Identified Critical Limitation:** Contrastive clauses account for 41.2% of errors in DistilBERT. Absolute positional embeddings cannot resolve clauses where relative distance flips polarity.
- **Output:** Mentors authorized **Week 7 Sprint: Engineering the #1 Champion DeBERTa-v3-Omni model and expanding to the 4-Platform Omni-Corpus**.

---

# Week 7 (CURRENT ACTIVE WEEK): Champion Architecture: DeBERTa-v3-Omni & Omni-Corpus Expansion (Days 43–45+)

### Day 43: Multi-Platform Omni-Corpus Assembly (453,878 Reviews)
- **Focus:** Eliminating single-domain grocery bias; multi-domain corpus engineering.
- **Actions:**
  - Assembled the **Universal Omni-Corpus** spanning 4 consumer domains:
    - Amazon Fine Food: 393,890 reviews (culinary, packaging, taste).
    - Flipkart Electronics: 15,000 reviews (hardware specs, battery, delivery).
    - Yelp Dining: 25,000 reviews (hospitality, staff demeanor, wait times).
    - IMDb Media: 20,000 reviews (narrative discourse, complex contrastive clauses).
    - **Total Corpus Scale: 453,878 reviews**.
  - Standardized unified schema across all platforms: `review_id`, `text`, `cleaned_text`, `score`, `sentiment`, `platform`, `year`.
  - Partitioned into 281,271 training samples and 72,846 unseen multi-platform holdout tests.
- **Output:** Unified corpus exported to `data/processed/omni_corpus.csv`.

### Day 44: DeBERTa-v3 Disentangled Attention & Mean+Max Pooling Head
- **Focus:** Advanced Transformer architecture implementation in PyTorch.
- **Actions:**
  - Initialized `microsoft/deberta-v3-base` (86M backbone parameters).
  - Implemented Disentangled Attention: decoupled content embeddings $\mathbf{c}_i$ and relative position embeddings $\mathbf{p}_{i|j}$, decomposing attention into 4 projection components:
    $$\text{Attention}(i, j) = \mathbf{c}_i \mathbf{W}_{q,c} \mathbf{W}_{k,c}^T \mathbf{c}_j^T + \mathbf{c}_i \mathbf{W}_{q,c} \mathbf{W}_{k,r}^T \mathbf{p}_{j|i}^T + \mathbf{p}_{i|j} \mathbf{W}_{q,r} \mathbf{W}_{k,c}^T \mathbf{c}_j^T$$
  - Engineered custom **Concatenated Mean + Max Pooling Head**:
    $$\mathbf{h}_{\text{pooled}} = [\text{Mean\_Pool}(\mathbf{h}_{1\dots L}) \,\|\, \text{Max\_Pool}(\mathbf{h}_{1\dots L})] \in \mathbb{R}^{1536}$$
    Projected through a 2-layer MLP (LayerNorm &rarr; Linear(1536, 768) &rarr; GELU &rarr; Dropout(0.1) &rarr; Linear(768, 2)).
  - Executed training pipeline (`src/training/train_deberta.py`) with 3.2x class-weighted cross-entropy.
- **Output:** Checkpoint saved to `models/deberta_omni_v1/`.

### Day 45 (TODAY): Champion Evaluation, Universal FAISS & Technical Defense Manual
- **Focus:** Canonical evaluation, full-scale vector index build, technical defense compilation.
- **Actions:**
  - Evaluated `DeBERTa-v3-Omni` on the 72,846 multi-platform holdout test set (`scripts/eval_canonical_test.py`):
    - **Accuracy:** **98.14%** (+2.32% over DistilBERT).
    - **Macro-F1:** **96.46%** (+1.42% over DistilBERT).
    - **Negative Recall:** **93.91%** (+3.79% over DistilBERT).
    - **False Positives:** **663** (**32.3% reduction** vs. DistilBERT's 980).
    - **ROC-AUC:** **0.9958**.
    - **Cross-Domain Generalization:** Amazon (98.3%), Flipkart (97.8%), Yelp (96.9%), IMDb (96.2%).
  - Built full universal FAISS index (`scripts/build_universal_faiss.py`) indexing 159,988 vectors in 245 MB memory.
  - Compiled publication-grade **ML Systems & AI Architecture Technical Defense Manual** (`ML_Systems_Technical_Defense_Manual.pdf`) via Chrome headless.
- **Output:**
  - DeBERTa-v3-Omni officially established as the **#1 Project Champion**.
  - All 13 production data assets synchronized across `public/data/` and `dist/data/`.
  - FastAPI daemon and Vite production frontend operating live.

---

## Final Production Benchmark Scorecard (Week 7 Frozen Metrics)

| Model Architecture | Parameters | Accuracy | Macro-F1 | Neg Recall | Neg Precision | False Positives | ROC-AUC | ECE (Calibrated) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Dummy Baseline** | 0 | 81.90% | 0.4502 | 0.00% | 0.00% | 13,178 | 0.5000 | N/A | Baseline |
| **Multinomial Naive Bayes** | 10k feat. | 85.60% | 0.8130 | 72.50% | 76.20% | 3,120 | 0.8840 | N/A | Baseline |
| **Logistic Regression (L2)** | 10k feat. | 90.45% | 0.8710 | 81.40% | 85.10% | 2,015 | 0.9325 | 0.0820 | Baseline |
| **Linear SVM (TF-IDF Tuned)** | 40k feat. | 95.61% | 0.9141 | 81.86% | 89.26% | 2,072 | 0.9823 | 0.0710 | Week 3 Milestone |
| **DistilBERT (281k Train)** | 66M | 95.82% | 0.9504 | 90.12% | 91.80% | 980 | 0.9882 | 0.0178 ($T=1.654$) | #2 Runner-Up |
| **DeBERTa-v3-Omni (#1 Champion)** | **86M** | **98.14%** | **96.46%** | **93.91%** | **94.80%** | **663 (-32.3%)** | **0.9958** | **0.0142** | **#1 Champion (Week 7)** |

---

*Log verified, locked, and compliant with DS Group MFD Internship Guidelines.*
