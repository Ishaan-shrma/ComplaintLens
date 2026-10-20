# Review Intelligence Agent — DS Group Internship

An AI-powered product review intelligence system that analyzes Amazon Fine Food Reviews to surface customer sentiment, product pain points, review quality signals, and trend data — built as part of a 45-day AI/ML Engineering internship at DS Group.

## Objective

- **Business:** explain customer sentiment, product pain points, review quality and trend signals from e-commerce review data.
- **Technical:** reproducible data cleaning, NLP preprocessing, supervised sentiment/rating models, model evaluation, vector search, and a dashboard.
- **Stretch:** a read-only AI review analyst agent that searches reviews semantically, answers approved questions, and explains results in plain language.

## Dataset

- **Name:** Amazon Fine Food Reviews
- **Source:** [Kaggle — snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Contents:** ~568K Amazon food-product reviews with user, product, rating, helpfulness, summary, full review text, and timestamp fields.

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/ds-group-review-intelligence.git
cd ds-group-review-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (requires a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d snap/amazon-fine-food-reviews -p data/raw --unzip

# 4. Confirm it loaded correctly
python -c "import pandas as pd; df = pd.read_csv('data/raw/Reviews.csv'); print(df.shape)"
```

## Project Structure

```
data/raw/          # Original Kaggle files (not committed — see .gitignore)
data/processed/    # Cleaned datasets and feature tables
notebooks/         # 01_data_understanding, 02_nlp_eda, 03_sentiment_model, 04_embeddings_retrieval
src/                # Reusable scripts (cleaning, features, training, retrieval)
models/             # Saved vectorizer, trained model, embedding index
dashboards/         # Streamlit app and dashboard exports
reports/            # Weekly summaries, final report, model card
presentations/      # Weekly office presentation slides and final deck
```

## Weekly Progress

### Week 1: Data Onboarding & EDA
- **Dataset cleaning:** 568,454 → 393,890 reviews (174,562 duplicates removed)
- **Features created:** `sentiment_label` (3-class: positive/neutral/negative), temporal features, helpfulness ratio
- **Initial EDA:** Rating distribution, review volume trends (1999-2012), length analysis, helpfulness patterns
- **Mentor feedback applied:** Dropped unnecessary columns, added word clouds, per-column statistics, CV framework

### Week 2: NLP Exploration
- **Text processing:** Built cleaning pipeline (lowercase, contraction expansion, stopword removal preserving negations), generated `cleaned_text`
- **N-gram analysis:** Bigrams and trigrams extracted by sentiment, `CountVectorizer(ngram_range=(1,3))`
- **Key finding:** Helpfulness correlation investigation — debunked length→quality myth (voting volume artifact, 0/0 cases excluded)
- **Business queries:** 16 SQL/Pandas queries for rating trends, product helpfulness, temporal sentiment shifts
- **Dashboard:** Streamlit prototype with KPIs, sentiment mix, interactive filtering

### Week 3: Baseline ML & Hyperparameter Tuning
- **Binary classification:** 291,297 train / 72,825 test (80/20 stratified split)
- **Baseline model:** Linear SVM + TF-IDF → 95.42% accuracy, 0.9109 macro-F1
- **Hyperparameter tuning:** 12 configurations, 2-fold CV (memory constraint)
  - Best config: `max_features=40000`, `ngram_range=(1,3)`, `C=0.5`
  - **Final performance:** 95.61% accuracy, 0.9141 macro-F1 (+0.19 pts gain)
- **Transformer comparison:** Classical ML beat zero-shot transformers by 10+ points
  - RoBERTa (Twitter): 85.94% accuracy
  - VADER (lexicon): 88.51% accuracy
  - Best classical: 95.61% accuracy
- **Error analysis:** False positives (mixed sentiment, sarcasm), false negatives (implicit recommendations)
- **Training time:** 7.1 minutes for full grid search

### Week 4: Advanced ML & Embeddings
**Completed:** October 1, 2026

#### Embedding System (Days 22-24)
- **Model:** `all-MiniLM-L6-v2` (384-dimensional sentence embeddings)
- **Sample Size:** 100,000 reviews (representative sample from 393K dataset)
- **Vector Index:** FAISS IndexIDMap with L2 distance
- **Search Features:**
  - Semantic similarity search
  - Sentiment-based filtering (positive/negative/neutral)
  - ID-based retrieval with O(1) metadata lookup

#### Topic Discovery (Day 25)
- **Method:** K-Means clustering on embeddings (k=5)
- **Topics Identified:**
  1. Coffee enthusiasts (premium coffee products)
  2. Tea lovers (tea varieties and accessories)
  3. Pet food buyers (dog/cat nutrition)
  4. Snack & candy shoppers (treats and sweets)
  5. Pasta & sauce fans (Italian cooking ingredients)
- **Visualization:** Topic distribution bar chart

#### Model Interpretability (Day 26)
- **SHAP Explanations:** Word-level importance with waterfall and bar plots
- **LIME Analysis:** Local feature importance for individual predictions
- **Attention Visualization:** Token-to-token attention heatmaps from DistilBERT
- **Insights:** Model focuses on product quality words, taste descriptors, and sentiment modifiers

#### Deep Learning Integration (Day 27)
- **Model:** DistilBERT fine-tuned on 281K reviews
- **Test Accuracy:** 97.39%
- **Integration:** Loaded into notebook 04 with prediction function
- **Combined Demo:** Semantic search + real-time sentiment prediction
- **Examples:** 5 diverse search scenarios (coffee, quality issues, pet food, price concerns, positive filter)

#### Deliverables
- `models/review_embeddings.npy` - 100K review embeddings (384-dim)
- `data/processed/embedding_metadata.csv` - Review metadata with IDs
- Notebook 04: Complete embedding pipeline with interpretability analysis
- FAISS index with semantic search API
- Topic clustering visualization

### Week 5: AI Assistant Prototype
_(AI assistant prototype)_

### Week 6: Final Walkthrough
_(final walkthrough)_

## Findings & Results

### Model Performance (Week 3)
- **Best Model:** Linear SVM with TF-IDF vectorization
- **Accuracy:** 95.61% on 72,825 test samples
- **Macro F1:** 0.9141 (balanced performance across classes)
- **Per-Class F1:** Positive 0.9742, Negative 0.8540
- **Training:** 7.1 minutes for hyperparameter grid search (12 configs, 2-fold CV)

### Key Insights (Week 3)
1. **Classical ML competitive:** TF-IDF + Linear SVM outperformed zero-shot transformers by 10+ percentage points
2. **Trigrams matter:** `ngram_range=(1,3)` captured longer phrases ("not worth the money" vs "worth the money")
3. **Hyperparameter gains:** 40K vocabulary + trigrams yielded +0.19 point improvement over baseline
4. **Data quality rigor:** Identified and excluded 0/0 helpfulness cases (undefined ratios treated as missing data)
5. **Imbalance handled:** 84.3% positive / 15.7% negative split, negative class still achieved 0.85 F1

### Deep Learning Performance (Week 4)
- **Model:** DistilBERT fine-tuned (6 layers, 66M parameters)
- **Training Samples:** 281,271 reviews
- **Test Accuracy:** 97.39% (+1.78 points over classical ML)
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Vector Index:** FAISS with 100K embeddings (~152MB)
- **Search Speed:** Sub-second semantic search on 100K reviews

### Key Insights (Week 4)
1. **Deep learning gains:** DistilBERT achieved 97.39% accuracy, outperforming Linear SVM (95.61%) by 1.78 points
2. **Semantic search works:** FAISS + sentence embeddings enable natural language queries ("coffee that tastes smooth")
3. **Topic clustering:** 5 distinct product categories emerged (coffee, tea, pet food, snacks, pasta) without supervision
4. **Interpretability patterns:** SHAP/LIME/Attention all identified quality adjectives (excellent, terrible, perfect, awful) as top predictors
5. **Attention focus:** Model attends to product-specific nouns (coffee, taste, flavor) and sentiment modifiers (not, very, really)
6. **False confidence:** Model predicts 99.9% confidence even on ambiguous reviews, suggesting over-fitting to training distribution

### Data Quality
- **Original size:** 568,454 reviews
- **After cleaning:** 393,890 reviews (69.3% retained)
- **Binary task:** 364,122 samples (positive/negative only)
- **Duplicates removed:** 174,562 (same UserId+Time+Text across products)
- **Timeline:** 1999-2012 Amazon Fine Food Reviews

## Running the Demo

### 1. Legacy Streamlit Exploration Platform
```bash
streamlit run dashboards/app.py
```

---
## Review Intelligence Prototype (React + Vite System)

An interactive, read-only multi-platform review intelligence application built strictly according to Rules R1–R8.

### Scope & Naming
- **Application Name:** Review Intelligence Prototype (Read-Only)
- **Coverage:** English-language review corpora across four consumer domains (Amazon Fine Food, Flipkart Retail, Yelp Dining, and IMDb Cinema).
- **Validation:** Prototype for evaluation and decision intelligence; not validated for autonomous production execution.
- **Champion Definition:** *"Best on the held-out test set among the models compared here"* (`DeBERTa-v3-Omni`, Macro-F1 = 0.9646, Neg Recall = 0.9391 on `amazon_test` $N = 72,846$).

### Reproducibility & Random Seed
- **Global Random Seed:** `42` (strictly applied across dataset stratification, training folds, DistilBERT fine-tuning, and evaluation partitions).

### Environment & Dependency Versions
- **Python Environment:**
  - `Python`: 3.10+ (tested on Python 3.12/3.14 with PyTorch 2.6.0 & CUDA 12.8)
  - `fastapi`: >= 0.115.0
  - `uvicorn`: >= 0.34.0
  - `torch`: >= 2.6.0
  - `transformers`: >= 4.49.0
  - `faiss-cpu`: >= 1.10.0
  - `pydantic`: >= 2.0.0
- **Node.js Environment:**
  - `Node.js`: >= 20.0.0
  - `npm`: >= 10.0.0
  - `react`: ^18.2.0
  - `react-dom`: ^18.2.0
  - `vite`: ^5.1.6
  - `tailwindcss`: ^3.4.1
  - `recharts`: ^2.12.2
  - `framer-motion`: ^11.0.8
  - `lucide-react`: ^0.358.0

### Exact Execution Order

Follow this step-by-step sequence to reproduce data contract generation, run integrity audits, verify AST literal rules, launch the inference backend, and build the frontend:

```bash
# Step 1: Generate all 13 Section 4 data contracts into public/data/
python scripts/prepare_data_contracts.py

# Step 2: Run strict Section 5 export integrity checks (6 key validation tests)
python scripts/verify_export.py

# Step 3: Run the AST metric literal linter (Rule R1: Single Source of Truth)
npm run check:literals

# Step 4: Launch the read-only FastAPI inference and agent backend
python server/app.py

# Step 5: (In a separate terminal) Launch the React + Vite frontend
npm run dev

# Step 6: Verify production build (executes check:literals before Vite bundle compilation)
npm run build
```

### Seven Workspaces
- `[01] Executive Overview` (`corpus.json`): Raw ingest, deduplication, invalid record removal, neutral quarantine, partition allocations, and per-platform class distributions.
- `[02] Champion: DeBERTa-v3-Omni` (`champion_deberta_metrics.json`, `deberta_ofat_trials.json`): Confusion matrix recomputations, multi-domain generalization, and validation-only OFAT studio.
- `[03] Benchmarks and Model Archive` (`model_comparison.json`, `distilbert_multiparams.json`, `training_curve.json`, `calibration_curve.json`): Like-with-like leaderboard deltas (R4), two linked DistilBERT facets, training loss curves, and precomputed reliability diagrams.
- `[04] Inference Lab` (`threshold_curve.json`, POST `/predict`): Live dual-model scoring, short input warnings, contrastive clause decomposition, validation decision threshold trade-off curves, and recorded fallbacks.
- `[05] Cross-Domain Analytics` (`omni_cross_domain_metrics.json`, `domain_vocab.json`): Cross-platform Macro-F1 and negative recall comparisons, label derivation methodologies, and distinctive domain vocabularies.
- `[06] AI Review Analyst` (`retrieval_eval.json`, `guardrails.json`): Auto-focused search bar, root-cause synthesis, KPI chips, recommendations, cited review cards with review IDs, and collapsible retrieval/guardrail audit logs.
- `[07] Error Taxonomy and Diagnostics` (`error_taxonomy.json`): Systematic categorization of 1,901 reviewed errors with horizontal bar distributions (`height={460}`, `YAxis width={250}`), audit shares, and sample review IDs.

### Engineering Rules Enforced (R1 to R8)
- **R1 (Single Source of Truth):** Zero hardcoded metric values in JSX; enforced by AST traversal linter (`scripts/check-literals.js`).
- **R2 (Context on Every Chart):** Model, evaluation set ID, split name, and sample size ($n$) printed on every card and chart.
- **R3 (File Wins Over Spec):** Verified artifact values take precedence over spec; differences logged to `DISCREPANCIES.md` (currently 0 discrepancies).
- **R4 (Like-with-Like Comparisons):** Deltas and rankings computed only between runs sharing the same `eval_set.id`. Rows evaluated on different sets are displayed greyed out.
- **R5 (Validation-Only Hyperparameter Selection):** OFAT studio, class-weight sweeps, and threshold curves tuned exclusively on validation data. Test metrics evaluated once on final canonical hold-out.
- **R6 (Careful Non-Causal Wording):** Measured correlations phrased as *"associated with"* rather than *"proves"* or *"drives"*.
- **R7 (Transparent Missing Data):** Absent keys or files render `"data file not found: <path>"` with no synthetic fallbacks.
- **R8 (Global Limitations):** Persistent drawer footer accessible from every workspace presenting the 7 engineering limitations.

