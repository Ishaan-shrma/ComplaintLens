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
_(advanced ML and embeddings)_

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

### Key Insights
1. **Classical ML competitive:** TF-IDF + Linear SVM outperformed zero-shot transformers by 10+ percentage points
2. **Trigrams matter:** `ngram_range=(1,3)` captured longer phrases ("not worth the money" vs "worth the money")
3. **Hyperparameter gains:** 40K vocabulary + trigrams yielded +0.19 point improvement over baseline
4. **Data quality rigor:** Identified and excluded 0/0 helpfulness cases (undefined ratios treated as missing data)
5. **Imbalance handled:** 84.3% positive / 15.7% negative split, negative class still achieved 0.85 F1

### Data Quality
- **Original size:** 568,454 reviews
- **After cleaning:** 393,890 reviews (69.3% retained)
- **Binary task:** 364,122 samples (positive/negative only)
- **Duplicates removed:** 174,562 (same UserId+Time+Text across products)
- **Timeline:** 1999-2012 Amazon Fine Food Reviews

## Running the Demo

```bash
streamlit run dashboards/app.py
```
