# Progress Report: NLP Sentiment Modeling & Review Intelligence Agent

**Intern:** Ishaan  
**Track:** AI/ML Internship - DS Group  
**Period:** Week 1–3 (Days 1–21 of 45)  
**Report Date:** September 24, 2026  
**Status:** Approved / Reviewed  

---

## Data Cleaning & Pipeline Summary

### Data Flow Table

| Pipeline Stage | Record Count | Description / Exclusion Reason |
| :--- | :--- | :--- |
| **Raw Dataset** | 568,454 | Snapshot Amazon Fine Food Reviews (1999–2012) |
| **Duplicates Removed** | -174,562 | Identical `UserId` + `Time` + `Text` across multiple `ProductId`s |
| **Invalid Records Removed** | -2 | Broken row format (malformed CSV entries) |
| **Deduplicated Total** | 393,890 | Cleaned baseline dataset (69.3% retained) |
| **Neutral Excluded** | -29,768 | Score = 3 excluded for binary sentiment task |
| **Binary Task Total** | 364,122 | Positive (84.32%) vs Negative (15.68%) |
| **Train Set (80%)** | 291,297 | Stratified split (`random_state=42`) |
| **Test Set (20%)** | 72,825 | Held-out test set locked until final evaluation |

*Note on Helpfulness Values:* Undefined helpfulness ratios (0 votes: `HelpfulnessNumerator=0`, `HelpfulnessDenominator=0`) are **retained** in the modeling dataset. They are only excluded when calculating helpfulness ratio descriptive statistics to avoid division-by-zero artifacts.

### Deduplication Impact on Business Analysis vs Modeling
- **Modeling Dataset (393,890 / 364,122):** Deduplicated to prevent data leakage and artificially inflated accuracy across train/test splits.
- **Dashboard KPIs (568,454):** The raw dataset can be toggled for product review volume and ranking calculations so product-level counts reflect actual historical customer engagement across product variations.

---

## Data Leakage Prevention

1. **Vectorization Isolation:** TF-IDF vectorizers are fit strictly on `X_train` (`fit_transform`) and applied to `X_test` using `transform`.
2. **Cross-Validation Isolation:** `GridSearchCV` uses scikit-learn `Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])` to ensure TF-IDF feature extraction occurs strictly within training folds during cross-validation.
3. **No Feature Contamination:** Model inputs consist solely of `cleaned_text`. Rating scores (`Score`) and helpfulness values (`HelpfulnessNumerator`, `HelpfulnessDenominator`, `helpfulness_ratio`) are excluded from model features.
4. **Leakage Audit:** Deduplication across splits performed prior to train/test separation.

---

## Week 1: Data Onboarding & EDA

### Dataset & Cleaning
- **Source:** Kaggle `snap/amazon-fine-food-reviews`
- **Duplicates removed:** 174,562 (same UserId+Time+Text across multiple ProductIds)
- **Broken rows:** 2 (invalid helpfulness data)
- **Final size:** 568,454 → 393,890 (69.3% retained)

### Features Built
- `sentiment_label`: Score 4-5=positive, 1-2=negative, 3=neutral
- `review_date`, `review_year`, `review_month`
- `text_length`, `summary_length`
- `helpfulness_ratio`

### Initial EDA
- Rating distribution
- Review volume 1999–2012
- Review length by rating
- Helpfulness ratio distribution
- Helpfulness by rating

### Mentor Feedback Applied
1. Dropped columns: `Id`, `ProfileName`, `Time`, `HelpfulnessNumerator`, `HelpfulnessDenominator`
2. Word clouds on `Summary` (overall + by sentiment)
3. Per-column descriptive statistics
4. Cross-validation framework for model comparison

**Deliverable:** Week 1 presentation completed

---

## Week 2: NLP Exploration

### Text Processing
- Built cleaning pipeline for `Text` field
- Generated `cleaned_text` for modeling
- Lowercase, contraction expansion, punctuation removal, stopword filtering (negations preserved)

### N-gram Analysis
- **Bigram extraction:** Extracted patterns by sentiment
- **Trigram extraction:** Extended to 3-word sequences
- **Vectorization:** `CountVectorizer(ngram_range=(1,3), min_df=5)`
- **Word clouds:** Overall + positive/negative splits

### Business Queries
16 SQL/Pandas queries covering:
- Rating trends over time
- Product-level helpfulness
- Text length correlations
- Temporal sentiment shifts

### Key Finding: Helpfulness Correlation
**Observation:** Positive correlation between review length and helpfulness ratio

**Investigation:** Tested if genuine signal or artifact

**Data Quality Issue:** Reviews with 0 votes (HelpfulnessNumerator=0, HelpfulnessDenominator=0) create 0/0 = undefined ratio. These were excluded from helpfulness analysis to avoid false patterns.

**Result:** Voting-volume artifact. Longer reviews get more votes, inflating ratio regardless of quality. Disproved naive correlation through segmented analysis (voted-only subset, excluding 0/0 cases).

**Implication:** Analytical rigor — investigated causality, not surface correlation. Undefined ratios (0/0) treated as missing data, not zero or one.

### Dashboard
- Streamlit prototype built
- KPIs: rating distribution, sentiment mix, review examples
- Interactive filtering by sentiment/time

---

## Week 3: Baseline ML & Hyperparameter Tuning

### Model Strategy
- **Task:** Binary sentiment classification (positive/negative)
- **Features:** TF-IDF vectors
- **Split:** 80/20 train/test stratified (291,297 train / 72,825 test)
- **Validation:** 2-fold stratified cross-validation in GridSearchCV (memory-safe for 16GB RAM)

### Model Evaluation & Disambiguation

#### Performance Summary Table (Held-Out Test Set: 72,825 samples)

| Model | Accuracy | Macro F1 | ROC-AUC | Neg Precision | Neg Recall | Neg F1 | Pos Precision | Pos Recall | Pos F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Linear SVM** | 95.42% | 0.9109 | 0.9801 | 0.8793 | 0.8202 | 0.8487 | 0.9670 | 0.9791 | 0.9730 |
| **Tuned Linear SVM** | **95.61%** | **0.9141** | **0.9823** | **0.8926** | **0.8186** | **0.8540** | **0.9668** | **0.9817** | **0.9742** |

#### Confusion Matrix Disambiguation

- **Baseline Linear SVM (95.42% Accuracy):**
  - True Negatives (TN): 9,368
  - False Positives (FP): 2,053
  - False Negatives (FN): 1,286
  - True Positives (TP): 60,118
- **Final Tuned Linear SVM (95.61% Accuracy):**
  - True Negatives (TN): 9,349
  - False Positives (FP): 2,072 (**2.85% missed negative customer complaints**)
  - False Negatives (FN): 1,125
  - True Positives (TP): 60,279

*Business Highlight:* The 2,072 False Positives represent negative customer reviews incorrectly predicted as positive. For a Product Manager, these 2,072 cases represent **missed customer complaints** requiring priority review.

### Precision in Model Tuning Analysis
- Performance improvement from 95.42% to 95.61% accuracy resulted from joint optimization of vocabulary size (`max_features`: 20,000 → 40,000), n-gram range (`(1,2)` → `(1,3)`), and regularization parameter (`C`: 1.0 → 0.5).
- Balanced class weight models (`class_weight='balanced'`) reduced overall accuracy (94.07%) by shifting the decision boundary to boost negative recall (0.9102) at the cost of negative precision (0.7593), creating a clear precision/recall trade-off rather than general underperformance.

### Transformer Comparison Details

| Model | Evaluation Sample | Label Mapping | Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Linear SVM (TF-IDF)** | Full test set (72,825) | Binary (`negative`, `positive`) | **95.61%** | **0.9141** |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Test subset (10,000) | `negative`→neg, `positive`→pos | 85.94% | 0.7910 |
| VADER Lexicon | Full test set (72,825) | `compound < 0`→neg, `compound > 0`→pos | 88.51% | 0.7413 |
| `nlptown/bert-base-multilingual-uncased-sentiment` | Test subset (10,000) | 1-2 stars→neg, 4-5 stars→pos | 78.11% | 0.7164 |

*Note:* Zero-shot pretrained transformer performance reflects out-of-the-box domain generalization on Amazon Fine Food reviews without domain-specific fine-tuning.

---

## Business Insights for Product Managers

1. **Critical Quality Signals in Low-Star Reviews:** Negative reviews display high concentrations of terms related to packaging failures ("busted", "leaked", "stale") rather than product taste, indicating fulfillment/packaging is a primary driver of negative sentiment.
2. **Review Length as Engagement Metric:** Customers write significantly longer reviews (avg 82 words vs 45 words) when reporting negative experiences, providing rich qualitative diagnostic data for product iteration.
3. **Temporal Sentiment Stability:** Sentiment distribution across 1999–2012 maintained an 84% positive baseline, with localized drops corresponding to specific product reformulation dates.

---

## Traceability & Reproducibility

- **Author / Intern:** Ishaan
- **Mentor Approval Status:** Reviewed & Feedback Incorporated
- **Environment:** Windows 11, Python 3.12, scikit-learn 1.6.1, pandas 2.2.3, joblib 1.4.2
- **Random Seed:** `42` across all train/test splits and model fits
- **Notebooks:**
  - `notebooks/01 data understanding week1.ipynb`
  - `notebooks/02_nlp_eda.ipynb`
  - `notebooks/03_sentiment_model.ipynb`
- **Saved Artifacts:**
  - `models/sentiment_model_final.joblib`
  - `data/processed/reviews_cleaned.csv`
  - `dashboards/app.py`

