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

- **Week 1:** _(fill in after Day 7 — dataset profile, data quality issues found, sentiment_label decision, first EDA insights)_
- **Week 2:** _(NLP EDA and model framing)_
- **Week 3:** _(baseline ML results)_
- **Week 4:** _(advanced ML and embeddings)_
- **Week 5:** _(AI assistant prototype)_
- **Week 6:** _(final walkthrough)_

## Findings & Results

_(fill in near the end — key insights, model performance, retrieval/agent capabilities)_

## Running the Demo

```bash
streamlit run dashboards/app.py
```
