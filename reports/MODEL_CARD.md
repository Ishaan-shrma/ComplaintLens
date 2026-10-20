# Model Card: Amazon Fine Food Review Sentiment Classifier

**Model Version:** 1.0  
**Date:** September 2026  
**Author:** Ishaan  
**Model Type:** Linear Support Vector Machine (LinearSVC) with TF-IDF features

---

## Model Description

### Intended Use
This sentiment classifier predicts whether Amazon Fine Food reviews are **positive** or **negative** based on review text. It is designed to:
- Automatically flag negative customer feedback for priority support response
- Analyze customer sentiment trends over time
- Identify product quality issues through complaint detection

**Primary Users:** Customer support teams, product managers, quality assurance teams

### Out-of-Scope Use Cases
- **NOT for moderation:** This model classifies sentiment, not toxicity or policy violations
- **NOT for user profiling:** Do not use to make decisions about individual customers
- **NOT for other domains:** Trained only on food product reviews; performance on other categories (electronics, books) is unknown
- **NOT real-time critical:** 4.39% error rate unsuitable for high-stakes automated decisions without human review

---

## Model Architecture

**Pipeline:**
```
Raw Review Text → TF-IDF Vectorization → Linear SVM → Binary Classification
```

**Hyperparameters (tuned via 2-fold CV):**
- `TfidfVectorizer`: max_features=40,000, ngram_range=(1,3), min_df=5
- `LinearSVC`: C=0.5, max_iter=2,000, random_state=42

**Training Data:**
- Dataset: Amazon Fine Food Reviews (Kaggle)
- Total cleaned samples: 364,122 reviews (after removing neutral ratings)
- Train/test split: 80/20 stratified split
- Training set: 291,297 reviews (84.3% positive, 15.7% negative)
- Test set: 72,825 reviews (held-out)

---

## Performance Metrics

### Overall Performance (Test Set)
| Metric | Value |
|--------|-------|
| **Accuracy** | 95.61% |
| **Macro F1-Score** | 0.9141 |
| **ROC-AUC** | 0.9823 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negative** | 0.8926 | 0.8186 | 0.8540 | 11,421 |
| **Positive** | 0.9668 | 0.9817 | 0.9742 | 61,404 |

### Confusion Matrix (Test Set: 72,825 reviews)
```
                    Predicted
                Negative  Positive
Actual Negative   9,349     2,072   ← Missed Complaints (FP)
       Positive   1,124    60,280
```

**Key Error Types:**
- **False Positives (FP): 2,072 (2.84%)** - Negative reviews misclassified as positive
  - **Business Impact:** Missed customer complaints, delayed support response
## Training Process

### Data Cleaning Pipeline
1. **Deduplication:** Removed 174,564 duplicate reviews based on (UserId, Time, Text)
2. **Data validation:** Removed 29,768 rows with invalid helpfulness ratios
3. **Neutral removal:** Filtered out 3-star reviews to create binary classification task
4. **Feature engineering:** Created `cleaned_text` from lowercased, stopword-removed review text

### Cross-Validation Strategy
- **Method:** 2-fold Stratified K-Fold
- **Leakage Prevention:** TF-IDF vectorizer refits on each fold's training portion (Pipeline approach)
- **Hyperparameter Search:** GridSearchCV over 8 configurations
- **Best CV Macro-F1:** 0.9113

### Leakage Checks Performed
1. ✓ No train/test index overlap
2. ✓ Deduplication before split
3. ✓ TF-IDF vocabulary fitted only on training data (via Pipeline)
4. ✓ Feature isolation: Model uses `cleaned_text` only, no target leakage

---

## Limitations and Risks

### Known Limitations
1. **Class Imbalance:** 84.3% positive, 15.7% negative
   - Model optimized for macro-F1 to balance both classes
   - Negative recall (81.86%) lower than positive recall (98.17%)

2. **Missed Complaints (2,072 FP):**
   - 18.1% of actual negative reviews are misclassified as positive
   - These reviews bypass complaint handling workflows
   - **Mitigation:** Manual review of borderline predictions (decision scores near 0)

3. **Domain-Specific:** Trained exclusively on food product reviews
4. **No Sarcasm/Irony Detection:** Linear model cannot capture complex linguistic nuances
5. **Temporal Drift:** Reviews span 1999-2012; language patterns may have shifted

### Ethical Considerations
- **Bias:** Model may perform differently across demographic groups
- **Transparency:** Predictions should show confidence scores and allow human override
- **Accountability:** False negatives (missed complaints) harm customer satisfaction
- **Privacy:** Model uses only review text; no personal identifiers retained

---

## Model Interpretation

**Top Positive Features:** "delicious", "love", "best", "perfect", "excellent", "great taste", "highly recommend", "favorite", "amazing", "fresh"

**Top Negative Features:** "terrible", "stale", "waste money", "horrible", "disgusting", "worst", "awful", "disappointed", "rancid", "inedible"

**Insight:** Model relies heavily on explicit sentiment words. Subtle complaints may be missed.

---

## Deployment Recommendations

1. **Human-in-the-loop:** Flag predictions with decision scores between -0.5 and +0.5
2. **Confidence thresholds:** Only auto-route reviews with |decision_score| > 1.0
3. **Monitoring:** Track FP rate weekly; retrain if it exceeds 3%
4. **When to Retrain:** Monthly drift detection or if accuracy falls below 94%

---

## Reproducibility

**Environment:** Python 3.11.9, scikit-learn 1.5.1, pandas 2.2.2, numpy 1.26.4

**Random Seeds:** train_test_split (42), LinearSVC (42), GridSearchCV CV (42)

**Artifacts:**
- Model: `models/sentiment_model_final.joblib`
- Training data: `data/processed/reviews_cleaned.csv`
- Notebook: `notebooks/03_sentiment_model.ipynb`

**Citation:** J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.

---

**Model Status:** ✓ Validated for pilot deployment with human oversight

- **False Negatives (FN): 1,124 (1.54%)** - Positive reviews misclassified as negative
  - **Business Impact:** Overly critical filtering, minimal harm

---

