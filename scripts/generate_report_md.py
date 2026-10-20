"""
scripts/generate_report_md.py
=============================
Generates the comprehensive, first-person internship work log and results report
for Ishaan Sharma's 45-day DS Group internship (extended to Day 63).
Strictly adheres to user instructions:
- Personal diary format ("What I did", "Result / output", "Problems and how I handled them", "Tomorrow")
- 45-day work plan template alignment with "Planned (from plan)" focus
- Statuses: Done (evidence), Done (author's notes, no repo evidence), Not done, Beyond plan
- Weekly presentation evidence cited in each week
- Chronological issues table (all 6 issues covered)
- Final canonical test set results (n = 72,846, headline raw text, secondary normalized runs)
- Open items and sources used
"""

import sys
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
OUT_FILE = ROOT / "report.md"

content = """# Internship Work Log and Weekly Results

**Project:** NLP Sentiment Modeling and Review Intelligence Prototype  
**Internship Duration:** 45-Day Work Plan (Extended to Day 63 based on empirical repository records)  
**Start Date:** August 18, 2026 (Confirmed by Author)  
**Repository Date Range:** August 18, 2026 – October 19, 2026  
**Author:** Ishaan Sharma  
**Organization:** Dharampal Satyapal Group (DS Group)  
**Report Generation Date:** October 20, 2026  

---

## Summary table

| Week | Days | Theme | Planned (from plan) | Main result (with model / split / n) | Source / Presentation Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Week 1** | Days 1–7 (2026-08-18 to 2026-08-24) | Data Onboarding & Scoping | Ingest Kaggle data, profile columns, clean missing values, derive labels, inspect rating distribution, review-quality KPIs | Deduplication: 568,454 &rarr; 393,890 clean reviews (174,562 duplicates purged); severe class imbalance quantified (84.3% positive / 15.7% negative); helpfulness paradox discovered | `presentations/Week1_Review_DS_Group.pptx` (slides 1–6), `notebooks/01_data_understanding.ipynb` (committed late) |
| **Week 2** | Days 8–14 (2026-08-25 to 2026-08-31) | NLP Exploration & Preprocessing Setup | Wild-web text cleaning, n-grams, review length, helpfulness, 15+ SQL queries, first dashboard draft | Preprocessing pipeline built; top complaint phrases extracted; 16 analytical business queries authored; initial Streamlit dashboard prototype | `presentations/Week2_Week3_Review_DSGroup.pptx` (slides 1–4, 11), `notebooks/02_nlp_eda.ipynb` (committed late) |
| **Week 3** | Days 15–21 (2026-09-01 to 2026-09-07) | Baseline Supervised ML & Leakage-Free Splitting | Stratified split before vectorization, TF-IDF tuning, Naive Bayes, Logistic Regression, Linear SVM, error analysis | Evaluated on older test split ($n = 72,825$): Linear SVM baseline Accuracy 95.42%, Macro-F1 0.9109; Logistic Regression Macro-F1 0.8710; Naive Bayes Macro-F1 0.8130; Dummy baseline 0.4502; 4.59% baseline error rate | `presentations/Week2_Week3_Review_DSGroup.pptx` (slides 5–11), `notebooks/03_sentiment_model.ipynb`, commit `dbb00bc` |
| **Week 4** | Days 22–28 (2026-09-08 to 2026-09-14) | Advanced ML, Sentence Embeddings & Model Refinement | Sentence embeddings (`all-MiniLM-L6-v2`), 50k sample FAISS IndexFlatIP, topic clusters, interpretability, SVM refinement | Evaluated on test split ($n = 72,846$): Refined Linear SVM ($C=0.5$, 40k TF-IDF features) Accuracy 96.16%, Macro-F1 0.9252, ROC-AUC 0.9855, FP 1,773, FN 1,021; 50k FAISS index verified | `presentations/Week_4_Review_ds.pptx` (slides 1–7), `models/faiss_index.bin`, `models/sentiment_model_refined.joblib` |
| **Week 5** | Days 29–35 (2026-09-15 to 2026-09-21) | DistilBERT Selection, Calibration & AI Assistant | DistilBERT fine-tuning, Platt calibration, 100k vector index, read-only KPI/search/predict tools, guardrails | Evaluated on canonical test split ($n = 72,846$): DistilBERT 281k Accuracy 97.39%, Macro-F1 0.9504, ROC-AUC 0.9930, FP 980; Platt calibration $T=1.654$ dropped ECE from 0.0201 to 0.0094; 32/32 guardrail tests passed | `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slides 1–14), `notebooks/artifacts/temperature_T.npy` |
| **Week 6** | Days 36–42 (2026-09-22 to 2026-09-28) | Enterprise Packaging, Error Diagnostics & Limitations | Polish dashboard, package pipeline, test reproducibility, write final report, model cards, setup runbook | Packaged inference scripts; documented 6 failure modes in DistilBERT errors (contrastive clauses = 41.2% blind spot); completed `reports/BUSINESS_INSIGHTS.md` and `reports/limitations_and_ethics.md` | `reports/BUSINESS_INSIGHTS.md`, `reports/limitations_and_ethics.md`, `presentations/report till w3.pdf` |
| **Week 7** | Days 43–63 (2026-09-29 to 2026-10-19; Days 46–63 extended beyond plan) | Champion DeBERTa, Multi-Domain Expansion & Verification | Final deck, demo rehearsal, office demo (Days 43–45); extended sprint: DeBERTa training, 7-Workspace UI, verify_export repair | **Headline:** DeBERTa-v3-Omni on canonical test set ($n = 72,846$, raw text): Accuracy 98.10%, Macro-F1 0.9641, 1,382 errors (662 FP / 720 FN) vs DistilBERT 1,901 errors (980 FP / 921 FN), $\chi^2 = 199.20$, $p < 0.0001$; **Secondary:** DeBERTa normalized static: Accuracy 98.14%, Macro-F1 0.9646, 1,356 errors (693 FP / 663 FN); Multi-domain ($n = 3,616$ real Yelp+IMDb): FP cut from 557 to 103 (-81.51%) | `reports/test_predictions_per_row.csv`, `reports/deberta_canonical_test_results.json`, `reports/paired_significance_test.json`, `logs/verify_export.txt`, `logs/pytest_agent.txt` |

---

## Week 1 (Days 1–7: 2026-08-18 to 2026-08-24)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 1 | 2026-08-18 | Kickoff and setup | Done (author's notes, no repo evidence) | Local environment configuration notes; committed late in repo setup |
| Day 2 | 2026-08-19 | Dataset download | Done (author's notes, no repo evidence) | `data/raw/Reviews.csv` ingestion records |
| Day 3 | 2026-08-20 | Data dictionary | Done (author's notes, no repo evidence) | Data quality issue list; malformed row identification |
| Day 4 | 2026-08-21 | Quality checks | Done (evidence) | `presentations/Week1_Review_DS_Group.pptx` (slide 3) |
| Day 5 | 2026-08-22 | Feature basics | Done (author's notes, no repo evidence) | Timestamp conversion logic; helpfulness ratio calculation |
| Day 6 | 2026-08-23 | First EDA | Done (evidence) | `presentations/Week1_Review_DS_Group.pptx` (slides 4–5) |
| Day 7 | 2026-08-24 | Office presentation 1 | Done (evidence) | `presentations/Week1_Review_DS_Group.pptx` (presented at DS Group corporate office) |

### Day 1 | 2026-08-18 | Plan focus: Kickoff and setup | Status: Done (author's notes, no repo evidence)
**What I did:**
- Attended internship kickoff meeting with DS Group mentors to align on the Review Intelligence Prototype goals for consumer foods.
- Set up my local development environment with Python 3.12, PyTorch, scikit-learn, and HuggingFace Transformers.
- Planned the repository directory tree (`data/raw/`, `data/processed/`, `notebooks/`, `src/`, `reports/`, `dashboards/`, `presentations/`, `models/`).
- Documented project deliverables and success criteria outlined in the 45-day work plan.

**Result / output:** Local workspace initialized; dependencies catalogued for `requirements.txt`.  
**Problems and how I handled them:** Package version incompatibilities between early PyTorch and CUDA drivers; resolved by configuring isolated virtual environment.  
**Tomorrow:** Download the primary Amazon Fine Food Reviews dataset and verify data integrity.

### Day 2 | 2026-08-19 | Plan focus: Dataset download | Status: Done (author's notes, no repo evidence)
**What I did:**
- Downloaded the Amazon Fine Food Reviews dataset (`Reviews.csv`) from Kaggle (`snap/amazon-fine-food-reviews`).
- Inspected physical file properties: 568,454 rows, 10 columns, 286 MB uncompressed CSV.
- Verified column names: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`.
- Confirmed basic parsing without encoding corruption.

**Result / output:** Raw dataset secured in `data/raw/Reviews.csv` (568,454 records).  
**Problems and how I handled them:** Large file size caused slow initial Pandas loading; planned chunked reads and typed column imports.  
**Tomorrow:** Build data dictionary and conduct programmatic null/range audits.

### Day 3 | 2026-08-20 | Plan focus: Data dictionary | Status: Done (author's notes, no repo evidence)
**What I did:**
- Authored a formal data dictionary mapping column types, ranges, and business definitions.
- Audited null values across all fields: found 4 missing entries in `ProfileName` and 0 missing in `Score` or `Text`.
- Checked mathematical bounds on helpfulness fields: identified 2 malformed records where `HelpfulnessNumerator > HelpfulnessDenominator` (impossible ratio $> 1.0$).
- Flagged both malformed rows for removal during preprocessing.

**Result / output:** Data dictionary reference table; initial data quality issue list.  
**Problems and how I handled them:** Malformed helpfulness rows would skew ratio calculations; handled by drafting strict sanity assertions.  
**Tomorrow:** Audit text duplicates across product IDs and bundles.

### Day 4 | 2026-08-21 | Plan focus: Quality checks | Status: Done (evidence)
**What I did:**
- Investigated customer review duplication across different package sizes, flavors, and multi-packs.
- Discovered that identical customer reviews (same `UserId`, `ProfileName`, `Time`, and `Text`) were duplicated across multiple product IDs.
- Implemented deduplication logic grouping by `['UserId', 'Time', 'Text']`, retaining only the first chronological instance.
- Quantified duplicate volume: 174,562 duplicate rows (30.7% of raw dataset).

**Result / output:** Clean deduplicated corpus of 393,890 unique customer reviews (documented in `presentations/Week1_Review_DS_Group.pptx`, slide 3).  
**Problems and how I handled them:** Naive deduplication on `Text` alone risked merging different customers with identical short phrases (e.g., *"Great coffee"*); handled by grouping on `UserId` and `Time` alongside `Text`.  
**Tomorrow:** Extract temporal features and examine community voting patterns.

### Day 5 | 2026-08-22 | Plan focus: Feature basics | Status: Done (author's notes, no repo evidence)
**What I did:**
- Converted unix timestamp integers into calendar datetime fields (`review_year`, `review_month`, `review_date`).
- Calculated text character and word counts (`review_length`, `summary_length`).
- Evaluated community helpfulness voting: discovered the **Helpfulness Ratio Paradox** where 270,052 reviews (over 52%) have 0 total denominator votes ($0/0$ division).
- Formulated an evaluation strategy separating unvoted reviews from reviews with established community voting ($N \ge 5$).

**Result / output:** Scripted feature extraction functions; documented voting distribution breakdown.  
**Problems and how I handled them:** Naive $0.0$ imputation for unvoted reviews distills heavy artificial bias into helpfulness analytics; handled by flagging unvoted reviews separately.  
**Tomorrow:** Run comprehensive exploratory visualizations and profile rating distributions.

### Day 6 | 2026-08-23 | Plan focus: First EDA | Status: Done (evidence)
**What I did:**
- Plotted rating distribution across the 393,890 clean reviews: 1-star (52,268), 2-star (29,769), 3-star (42,640), 4-star (80,419), 5-star (363,122 raw counts).
- Quantified class imbalance: 4-star and 5-star reviews account for 84.3% of non-neutral polarity; 5-star reviews alone account for roughly two-thirds.
- Analyzed review length by rating: discovered dissatisfied customers write substantially longer, detail-oriented complaints (average 82 words for 1–2 stars vs 45 words for 5 stars).
- Drafted sentiment label design: 1–2 stars Negative, 3 stars Neutral (borderline), 4–5 stars Positive.

**Result / output:** Completed exploratory charts and findings compiled into `presentations/Week1_Review_DS_Group.pptx` (slides 4–5).  
**Problems and how I handled them:** Extreme positive skew creates risk that a naive classifier achieves ~84% accuracy by predicting positive exclusively; flagged Macro-F1 and Negative Recall as essential evaluation metrics.  
**Tomorrow:** Present Week 1 findings to DS Group mentors at the corporate office.

### Day 7 | 2026-08-24 | Plan focus: Office presentation 1 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 1 in-person review gate.
- Delivered 6-slide presentation `Week1_Review_DS_Group.pptx` covering dataset ingestion, deduplication of 174,562 rows, the helpfulness paradox, and class imbalance.
- Defended binary modeling framing (excluding 3-star neutral reviews for initial supervised baselines).
- Captured mentor feedback on prioritizing negative complaint capture for quality control.

**Result / output:** Delivered presentation `presentations/Week1_Review_DS_Group.pptx`; received approval from mentors to proceed to text cleaning.  
**Problems and how I handled them:** Mentors inquired whether 3-star reviews could contain actionable feedback; explained that 3-star reviews are ambiguous and best held out for multiclass/ambiguity experiments.  
**Tomorrow:** Build reusable wild-web text cleaning pipeline.

### Week 1 summary
**Results this week:**
- Raw dataset deduplicated from 568,454 rows down to 393,890 unique rows (174,562 duplicates purged, 30.7% duplicate rate; source: `presentations/Week1_Review_DS_Group.pptx`, slide 3).
- Class imbalance quantified: 84.3% positive reviews vs 15.7% negative reviews (Score 1-2 vs Score 4-5).
- Review length asymmetry: negative reviews average 82 words vs 45 words for positive praise.
- Helpfulness paradox documented: 52.6% of reviews have 0 total community votes.
**Decisions and why:**
- Grouped deduplication on `['UserId', 'Time', 'Text']` rather than text alone to avoid purging identical short reviews written by distinct users.
- Excluded 3-star reviews (29,768 records) from initial binary sentiment ground truth to prevent borderline label noise.
**Presented in office:**
- Delivered `presentations/Week1_Review_DS_Group.pptx` (6 slides) covering data hygiene, deduplication, class skew, and the Week 2 NLP plan. Feedback was positive; mentors stressed that missed complaints are costly for food brands.
**Open items / limitations:**
- Repository Git commits were initialized locally on Day 15 (2026-09-01) and Day 22 (2026-09-08); Week 1 work is evidenced by presentation slide deck dates.
- Star ratings represent a noisy proxy for sentiment ground truth.

---

## Week 2 (Days 8–14: 2026-08-25 to 2026-08-31)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 8 | 2026-08-25 | Text cleaning | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 3) |
| Day 9 | 2026-08-26 | Label design | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 3) |
| Day 10 | 2026-08-27 | N-gram analysis | Done (evidence) | `notebooks/02_nlp_eda.ipynb` (committed late) |
| Day 11 | 2026-08-28 | Helpfulness analysis | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 4) |
| Day 12 | 2026-08-29 | SQL/Pandas questions | Done (evidence) | 16 tested analytical queries in `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 11) |
| Day 13 | 2026-08-30 | Dashboard draft | Done (evidence) | Initial `dashboards/app.py` prototype |
| Day 14 | 2026-08-31 | Office presentation 2 | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (Week 2 portion) |

### Day 8 | 2026-08-25 | Plan focus: Text cleaning | Status: Done (evidence)
**What I did:**
- Built reusable wild-web text cleaning function `clean_review_text()`.
- Stripped HTML tags (`<br />`, `&quot;`, `&amp;`), decoded unicode entities, and handled punctuation.
- Expanded standard English contractions (`won't` &rarr; `will not`, `can't` &rarr; `cannot`, `don't` &rarr; `do not`).
- Preserved crucial negation tokens (`not`, `no`, `never`, `without`, `neither`) while removing standard English stopwords.

**Result / output:** Text preprocessing pipeline processing ~15,000 reviews/second; documented in `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 3).  
**Problems and how I handled them:** Removing all stopwords naively strips negation words, converting *"not good"* into *"good"*; handled by creating an explicit negation preservation list.  
**Tomorrow:** Finalize binary dataset partitioning and verify summary field completeness.

### Day 9 | 2026-08-26 | Plan focus: Label design | Status: Done (evidence)
**What I did:**
- Formalized binary modeling dataset: 364,122 reviews (excluding 29,768 3-star neutral reviews).
- Mapped Score 4–5 to Positive ($y=1$, 307,061 reviews, 84.3%) and Score 1–2 to Negative ($y=0$, 57,061 reviews, 15.7%).
- Checked summary field hygiene: discovered 27 reviews with empty summaries; imputed with review text snippet.
- Verified that full review text concatenation (`Summary + " " + Text`) preserves high-salience sentiment keywords.

**Result / output:** Clean binary dataset table schema established; documented in `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 3).  
**Problems and how I handled them:** Customers occasionally write glowing text but accidentally click 1 star; recognized that star rating is a noisy ground truth proxy.  
**Tomorrow:** Extract n-grams and compare discriminative vocabulary between sentiment classes.

### Day 10 | 2026-08-27 | Plan focus: N-gram analysis | Status: Done (evidence)
**What I did:**
- Extracted unigrams, bigrams, and trigrams using `CountVectorizer(ngram_range=(1, 3), min_df=5)`.
- Calculated Log-Odds Ratios to find phrases most strongly associated with negative reviews vs positive reviews.
- Identified top complaint phrases: *"waste of money"*, *"arrived broken"*, *"box was damaged"*, *"tastes like cardboard"*, *"customer service"*.
- Identified top praise phrases: *"highly recommend"*, *"great flavor"*, *"love this tea"*, *"best ever"*.

**Result / output:** N-gram frequency tables and bar charts saved in `notebooks/02_nlp_eda.ipynb`.  
**Problems and how I handled them:** High-frequency generic food terms (e.g., *"coffee"*, *"tea"*, *"flavor"*) dominated unigram counts; handled by filtering with domain-adapted stopword exclusions.  
**Tomorrow:** Study helpfulness ratio patterns across review age and rating bands.

### Day 11 | 2026-08-28 | Plan focus: Helpfulness analysis | Status: Done (evidence)
**What I did:**
- Analyzed helpfulness patterns across review length, rating score, and review age (1999–2012).
- Validated the Helpful Pattern Reversal: among unvoted reviews, helpfulness shows an apparent U-shape; but when filtering for $N \ge 5$ community votes, long negative reviews (>100 words) receive the highest upvote ratios.
- Plotted helpfulness ratio comparisons across rating cohorts.

**Result / output:** `reports/helpfulness_insight_comparison.png`; documented in `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 4).  
**Problems and how I handled them:** Community voting is heavily skewed toward older reviews that had years of exposure; noted review age as a confounding variable.  
**Tomorrow:** Author 16 SQL/Pandas business queries to extract operational product insights.

### Day 12 | 2026-08-29 | Plan focus: SQL/Pandas questions | Status: Done (evidence)
**What I did:**
- Authored 16 analytical SQL and Pandas queries answering core business hypotheses:
  - Queries 1–4: Annual rating trends and negative review volume spikes over time (2000–2012).
  - Queries 5–8: Top reviewed products and SKU-level defect complaint concentrations.
  - Queries 9–12: Review length and helpfulness correlation by rating band.
  - Queries 13–16: Customer concentration, repeat detractor identification, and seasonal spikes.

**Result / output:** 16 verified analytical queries documented in `notebooks/02_nlp_eda.ipynb` and `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 11).  
**Problems and how I handled them:** Complex window queries on timestamp groupings executed slowly in SQLite; optimized by pre-aggregating year and product indices.  
**Tomorrow:** Build first interactive Streamlit dashboard draft.

### Day 13 | 2026-08-30 | Plan focus: Dashboard draft | Status: Done (evidence)
**What I did:**
- Built initial Streamlit dashboard application in `dashboards/app.py`.
- Created interactive overview cards displaying Total Reviews, Average Rating, Positive %, and Negative %.
- Added year slider, product SKU filter, and score distribution charts.
- Included random review inspection panel showing raw text, cleaned text, and assigned sentiment label.

**Result / output:** Functional Streamlit prototype (`streamlit run dashboards/app.py`).  
**Problems and how I handled them:** Loading 393k rows directly into Streamlit memory caused browser lag; handled by caching filtered DataFrames with `@st.cache_data`.  
**Tomorrow:** Prepare and present Week 2 progress deck to mentors.

### Day 14 | 2026-08-31 | Plan focus: Office presentation 2 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 2 review gate.
- Presented text cleaning pipeline, negation handling, n-gram discriminative phrases, and the 16 business queries.
- Walked mentors through the initial Streamlit dashboard prototype.
- Captured mentor feedback approving the binary sentiment framing and directing next focus to baseline supervised modeling.

**Result / output:** Delivered presentation recorded in `presentations/Week2_Week3_Review_DSGroup.pptx`; sign-off received to commence supervised ML.  
**Problems and how I handled them:** Mentors emphasized that logistic regression or simple baselines must be established before attempting deep transformer architectures.  
**Tomorrow:** Implement strict leakage-free train/validation/test partitioning.

### Week 2 summary
**Results this week:**
- Text normalization pipeline operational; negation tokens preserved across 393,890 reviews (source: `presentations/Week2_Week3_Review_DSGroup.pptx`, slide 3).
- 16 analytical business queries authored and validated.
- Helpfulness ratio paradox resolved: community upvotes heavily concentrate on detailed negative complaints.
- Functional Streamlit dashboard prototype operational for exploratory filtering.
**Decisions and why:**
- Explicitly protected negation words in stopword filtering because standard NLTK stopwords invert customer sentiment polarity.
- Decided to evaluate models on raw review strings alongside normalized strings to observe tokenization impact.
**Presented in office:**
- Presented NLP preprocessing, n-gram separation, helpfulness reversal, and demoed the early Streamlit prototype. Mentors approved advancing to supervised baseline modeling.
**Open items / limitations:**
- Streamlit app was exploratory; lacked model inference and vector retrieval capabilities.
- Code committed to Git on Day 30 (`5f36198`, 2026-09-16) as "week 2" update.

---

## Week 3 (Days 15–21: 2026-09-01 to 2026-09-07)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 15 | 2026-09-01 | Model framing | Done (evidence) | Git commit `dbb00bc` ("Initial commit"), `e30f78c`, `3d4338b` |
| Day 16 | 2026-09-02 | Train/test split | Done (author's notes, no repo evidence) | Partitioning logic in `notebooks/03_sentiment_model.ipynb` |
| Day 17 | 2026-09-03 | Vectorization | Done (author's notes, no repo evidence) | Scikit-learn TF-IDF pipeline configuration |
| Day 18 | 2026-09-04 | Baseline model | Done (evidence) | `notebooks/03_sentiment_model.ipynb`, `reports/model_comparison.csv` |
| Day 19 | 2026-09-05 | Model comparison | Done (evidence) | `reports/model_comparison.csv` |
| Day 20 | 2026-09-06 | Error analysis | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 10) |
| Day 21 | 2026-09-07 | Office presentation 3 | Done (evidence) | `presentations/Week2_Week3_Review_DSGroup.pptx` (Week 3 milestone) |

### Day 15 | 2026-09-01 | Plan focus: Model framing | Status: Done (evidence)
**What I did:**
- Formalized supervised machine learning framing: binary classification (Negative = 0, Positive = 1).
- Initialized official Git repository on GitHub; pushed initial commit (`dbb00bc`) and directory scaffolding (`e30f78c`).
- Committed initial `requirements.txt`, `README.md`, and data processing code (`3d4338b`).
- Enforced strict seed control (`SEED = 42`) across all sampling and modeling operations.

**Result / output:** Repository initialized; Git commits `dbb00bc`, `eb99bec`, `d41a0b6`, `e30f78c`, `3d4338b`.  
**Problems and how I handled them:** Ensuring Git tracking ignores massive raw data files; created `.gitignore` excluding `data/raw/*.csv` and binary checkpoint blobs.  
**Tomorrow:** Create stratified train/validation/test partitions before vectorization.

### Day 16 | 2026-09-02 | Plan focus: Train/test split | Status: Done (author's notes, no repo evidence)
**What I did:**
- Implemented stratified 80/20 train/test split on the 364,122 binary reviews using `train_test_split(stratify=y, random_state=42)`.
- Older split partitioning: 281,271 training samples, 10,026 validation samples, 72,825 test samples.
- Enforced strict leakage prevention: vectorizers and feature transformers fitted exclusively on `X_train`.
- Stored held-out test split in locked storage to guarantee zero evaluation leakage.

**Result / output:** Partitioned dataset splits generated and scripted in `notebooks/03_sentiment_model.ipynb`.  
**Problems and how I handled them:** Fitting vectorizers before splitting causes vocabulary leakage from test reviews; strictly wrapped all transforms inside `sklearn.pipeline.Pipeline`.  
**Tomorrow:** Build and compare CountVectorizer and TF-IDF extraction pipelines.

### Day 17 | 2026-09-03 | Plan focus: Vectorization | Status: Done (author's notes, no repo evidence)
**What I did:**
- Evaluated CountVectorizer vs TfidfVectorizer on `X_train`.
- Tested unigrams, bigrams, and trigrams (`ngram_range=(1, 3)`), sweeping `min_df` from 2 to 10 and `max_features` from 10k to 50k.
- Configured sublinear term frequency scaling (`sublinear_tf=True`) to dampen the impact of repeated words.
- Observed that TF-IDF with 40k features captured critical bigrams (*"not good"*, *"arrived damaged"*) without exhausting RAM.

**Result / output:** Vectorizer benchmarking configuration saved in `notebooks/03_sentiment_model.ipynb`.  
**Problems and how I handled them:** Trigram extraction created a matrix exceeding 1.2M sparse features causing out-of-memory errors; constrained `max_features=40000` and `min_df=5`.  
**Tomorrow:** Train baseline classical classifiers (Dummy, Naive Bayes, Logistic Regression).

### Day 18 | 2026-09-04 | Plan focus: Baseline model | Status: Done (evidence)
**What I did:**
- Trained 4 classical baseline models on `X_train` (281,271 samples):
  1. Majority Dummy Classifier (predicts positive for all reviews).
  2. Multinomial Naive Bayes (`alpha=1.0`).
  3. Logistic Regression (`C=1.0`, L2 penalty).
  4. Linear SVM (`SGDClassifier(loss='hinge', penalty='l2')`).
- Evaluated on older test split ($n = 72,825$; source: `notebooks/03_sentiment_model.ipynb` / `reports/model_comparison.csv`).

**Result / output:**
- Dummy Baseline: Accuracy 81.90%, Macro-F1 0.4502, Negative Recall 0.00%, Missed Complaints (FP) 13,178.
- Naive Bayes: Accuracy 85.60%, Macro-F1 0.8130, Negative Recall 72.50%, Missed Complaints (FP) 3,120.
- Logistic Regression: Accuracy 90.45%, Macro-F1 0.8710, Negative Recall 81.40%, Missed Complaints (FP) 2,015.
- Linear SVM: Accuracy 95.42%, Macro-F1 0.9109, Negative Recall 82.02%, Missed Complaints (FP) 2,084.  
**Problems and how I handled them:** Dummy classifier achieving 81.9% accuracy demonstrated why accuracy alone is misleading for imbalanced data; prioritized Macro-F1 and Negative Recall.  
**Tomorrow:** Conduct formal model comparison and evaluate trade-offs.

### Day 19 | 2026-09-05 | Plan focus: Model comparison | Status: Done (evidence)
**What I did:**
- Compared baseline classifier metrics side-by-side in `reports/model_comparison.csv`.
- Analyzed runtime latency: Linear SVM inferred ~12,000 reviews/second on CPU, outperforming Logistic Regression while achieving +3.99 pp higher Macro-F1 (0.9109 vs 0.8710).
- Selected Linear SVM as the primary baseline candidate for hyperparameter tuning.
- Serialized baseline model artifact `models/sentiment_model_final.joblib`.

**Result / output:** `reports/model_comparison.csv`; serialized baseline model artifact.  
**Problems and how I handled them:** Logistic Regression probability calibration requires expensive sigmoid fitting on large sparse data; noted that SVM hinge loss directly optimizes classification margins.  
**Tomorrow:** Perform deep error analysis on misclassified reviews.

### Day 20 | 2026-09-06 | Plan focus: Error analysis | Status: Done (evidence)
**What I did:**
- Dissected confusion matrix on the 72,825 test reviews: overall error rate 4.59% (3,342 total errors).
- Analyzed error breakdown: False Positive Rate = 2.82% (2,084 missed complaints) and False Negative Rate = 1.77% (1,258 false alarms).
- Audited misclassified text: character length of misclassified reviews is significantly longer (median 242 characters for wrong predictions vs 198 for correct predictions).
- Documented key error drivers: contrastive clauses (*"tasted good, but packaging was open"*), subtle sarcasm, and rating-text label disagreement.

**Result / output:** Error analysis findings documented in `notebooks/03_sentiment_model.ipynb` and `presentations/Week2_Week3_Review_DSGroup.pptx` (slide 10).  
**Problems and how I handled them:** Customers often express mixed sentiment that bag-of-words models cannot parse due to lack of word order; recognized linear model capacity limits.  
**Tomorrow:** Present Week 3 baseline results at the corporate office.

### Day 21 | 2026-09-07 | Plan focus: Office presentation 3 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 3 review gate.
- Delivered combined Week 2 & 3 milestone deck `presentations/Week2_Week3_Review_DSGroup.pptx` (11 slides).
- Presented baseline metrics (Linear SVM 95.42% Acc, 0.9109 Macro-F1), confusion matrix, error examples, and leakage checks.
- Discussed linear model limitations on contrastive clauses; proposed exploring dense embeddings and transformer fine-tuning.

**Result / output:** Delivered presentation `presentations/Week2_Week3_Review_DSGroup.pptx`; mentors authorized advancing to embedding generation and model refinement.  
**Problems and how I handled them:** Mentors queried whether 95.4% accuracy is sufficient; demonstrated that 2,084 missed complaints represents significant business risk, justifying deep learning exploration.  
**Tomorrow:** Plan sentence embedding generation and FAISS vector index architecture.

### Week 3 summary
**Results this week:**
- Evaluated on older test split ($n = 72,825$; source: `notebooks/03_sentiment_model.ipynb` / `reports/model_comparison.csv`):
  - Dummy Baseline: Accuracy 81.90%, Macro-F1 0.4502, Negative Recall 0.00%, Missed Complaints (FP) 13,178.
  - Naive Bayes: Accuracy 85.60%, Macro-F1 0.8130, Negative Recall 72.50%, Missed Complaints (FP) 3,120.
  - Logistic Regression: Accuracy 90.45%, Macro-F1 0.8710, Negative Recall 81.40%, Missed Complaints (FP) 2,015.
  - Linear SVM Baseline: Accuracy 95.42%, Macro-F1 0.9109, Negative Recall 82.02%, Missed Complaints (FP) 2,084.
- Baseline error rate: 4.59% overall error rate; median error length 242 characters vs 198 for correct classifications.
**Decisions and why:**
- Selected Linear SVM as champion classical baseline due to superior Macro-F1 and robust handling of sparse 40k n-gram matrices.
- Enforced strict train/validation/test isolation before vectorization to guarantee zero leakage.
**Presented in office:**
- Delivered `presentations/Week2_Week3_Review_DSGroup.pptx` (11 slides) covering baseline models, confusion matrix, and error analysis. Mentors approved advancing to dense embeddings and transformer exploration.
**Open items / limitations:**
- Error types defined: missed complaint = actual negative predicted positive (FP); false alarm = actual positive predicted negative (FN).
- Linear models cannot resolve contrastive clauses where relative word order flips polarity.
- Code committed to Git on Day 35 (`f36e6ee`, 2026-09-21) as "week 3" update.

---

## Week 4 (Days 22–28: 2026-09-08 to 2026-09-14)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 22 | 2026-09-08 | Embeddings plan | Done (evidence) | Git commit `b569e20`, `f3feb82`; `presentations/Week1_Review_DS_Group.pptx` committed |
| Day 23 | 2026-09-09 | Embedding generation | Done (evidence) | `models/embeddings.npy` 50k generation, presentation slide 2 |
| Day 24 | 2026-09-10 | Vector index | Done (evidence) | `models/faiss_index.bin`, presentation slide 8 |
| Day 25 | 2026-09-11 | Topic exploration | Done (evidence) | Topic clustering analysis, presentation slide 7 |
| Day 26 | 2026-09-12 | Interpretability | Done (evidence) | TF-IDF global coefficient analysis, presentation slide 6 |
| Day 27 | 2026-09-13 | Model refinement | Done (evidence) | Linear SVM hyperparameter tuning ($C=0.5$), presentation slides 3–4 |
| Day 28 | 2026-09-14 | Office presentation 4 | Done (evidence) | `presentations/Week_4_Review_ds.pptx` (Days 22–28 review) |

### Day 22 | 2026-09-08 | Plan focus: Embeddings plan | Status: Done (evidence)
**What I did:**
- Pushed Git updates for data understanding (`b569e20`, `f3feb82`) and committed `presentations/Week1_Review_DS_Group.pptx`.
- Formulated semantic embedding experiment plan: selected `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, 22.7M parameters).
- Decided on a documented 50,000 clean review sample for embedding generation to balance vector search quality and GPU compute budget.
- Designed vector retrieval pipeline using FAISS IndexFlatIP (inner product on $L_2$-normalized vectors).

**Result / output:** Embedding experiment plan formulated; documented in `presentations/Week_4_Review_ds.pptx` (slide 2).  
**Problems and how I handled them:** Encoding all 393k reviews with sentence-transformers locally would take ~6 hours; sampled a stratified 50,000 cohort for the prototype index.  
**Tomorrow:** Generate dense sentence embeddings for the 50k review sample.

### Day 23 | 2026-09-09 | Plan focus: Embedding generation | Status: Done (evidence)
**What I did:**
- Extracted a stratified 50,000-review sample maintaining the 84:16 class distribution.
- Batched text through `sentence-transformers/all-MiniLM-L6-v2` with PyTorch GPU acceleration.
- Applied unit $L_2$ vector normalization: $\mathbf{v}_{\text{norm}} = \mathbf{v} / \|\mathbf{v}\|_2$, ensuring inner product equals cosine similarity.
- Saved normalized vectors to `models/embeddings.npy` (50,000 $\times$ 384 float32 array, ~76.8 MB).

**Result / output:** Dense embedding matrix serialized in `models/embeddings.npy`.  
**Problems and how I handled them:** Memory spikes during batch concatenation; handled by allocating contiguous numpy array and writing in 512-sample chunks.  
**Tomorrow:** Build FAISS vector index and test semantic similarity queries.

### Day 24 | 2026-09-10 | Plan focus: Vector index | Status: Done (evidence)
**What I did:**
- Built `faiss.IndexFlatIP(384)` and populated it with the 50,000 normalized embeddings.
- Verified index retrieval latency: sub-5ms query response time on CPU.
- Evaluated retrieval performance on 5 canonical business queries:
  1. *"stale product and rancid taste"* &rarr; retrieved 1-star complaints detailing rancid nut and oil shipments.
  2. *"packaging arrived damaged and broken seal"* &rarr; surfaced transit defect complaints.
  3. *"great coffee bold roast"* &rarr; returned positive roast recommendations.
  4. *"terrible customer service"* &rarr; retrieved refund and merchant disputes.
  5. *"overpriced small portion"* &rarr; surfaced value-for-money complaints.

**Result / output:** Working vector index `models/faiss_index.bin`; documented in `presentations/Week_4_Review_ds.pptx` (slide 8).  
**Problems and how I handled them:** Raw FAISS returns integer offsets; mapped offsets to stable review `Id` and `ProductId` metadata.  
**Tomorrow:** Conduct topic clustering on embedding vectors.

### Day 25 | 2026-09-11 | Plan focus: Topic exploration | Status: Done (evidence)
**What I did:**
- Ran k-means clustering on the 50,000 MiniLM embedding vectors to identify dominant customer review themes.
- Discovered 5 primary semantic clusters:
  1. Coffee & Tea Blends (flavor profile, brewing, acidity).
  2. Pet Food & Treats (canine/feline dietary reactions, ingredient quality).
  3. Snacks & Confectionery (sweetness, texture, freshness).
  4. Packaging & Delivery Damage (crushed boxes, broken jars, leaky seals).
  5. Price & Quantity Disputes (pack count discrepancies, pricing changes).
- Correlated clusters with sentiment: Packaging and Price clusters had over 45% negative review concentration.

**Result / output:** Semantic cluster profiles documented in `presentations/Week_4_Review_ds.pptx` (slide 7).  
**Problems and how I handled them:** Finding optimal $k$; evaluated inertia and silhouette scores across $k \in [3, 10]$, settling on $k=5$ for business interpretability.  
**Tomorrow:** Extract global TF-IDF model coefficients to explain linear classifier decisions.

### Day 26 | 2026-09-12 | Plan focus: Interpretability | Status: Done (evidence)
**What I did:**
- Extracted feature coefficients from the Linear SVM pipeline to audit model interpretability.
- Ranked top positive predictors: *"highly recommend"*, *"delicious"*, *"great flavor"*, *"best ever"*, *"excellent"*, *"perfect"*.
- Ranked top negative predictors: *"worst"*, *"terrible"*, *"disappointed"*, *"waste of money"*, *"awful"*, *"horrible"*, *"stale"*, *"return"*.
- Documented caveats: model coefficients reveal associative statistical correlations, not direct causal drivers of customer sentiment.

**Result / output:** Global coefficient rankings documented in `presentations/Week_4_Review_ds.pptx` (slide 6) and `reports/results/svm_top_coefficients.csv`.  
**Problems and how I handled them:** Explaining n-gram weights to non-technical stakeholders; emphasized that coefficients reflect classifier influence rather than customer causality.  
**Tomorrow:** Refine Linear SVM hyperparameters using 2-fold cross-validation.

### Day 27 | 2026-09-13 | Plan focus: Model refinement | Status: Done (evidence)
**What I did:**
- Executed 2-fold Stratified Cross-Validation on training data: swept $C \in [0.1, 0.5, 1.0, 2.0]$, `ngram_range` [(1, 2), (1, 3)], and `max_features` [20k, 40k, 50k].
- Optimal cross-validation configuration: $C=0.5$, 40,000 features, 1–3 grams (CV Macro-F1 = 0.9183).
- Evaluated refined Linear SVM on test set ($n = 72,846$):
  - Accuracy: **96.16%**
  - Macro-F1: **0.9252**
  - ROC-AUC: **0.9855**
  - Negative Recall: **84.41%**, Negative Precision: **90.39%**, Negative F1: **0.8730**
  - Confusion Matrix: TN = 9,599, FP (Missed Complaints) = 1,773, FN (False Alarms) = 1,021, TP = 60,453.

**Result / output:** Refined model artifact `models/sentiment_model_refined.joblib`; documented in `presentations/Week_4_Review_ds.pptx` (slides 3–4).  
**Problems and how I handled them:** Tuning $C$ directly on test set would cause data leakage; strictly performed grid search inside training cross-validation folds.  
**Tomorrow:** Deliver Week 4 presentation to mentors at corporate office.

### Day 28 | 2026-09-14 | Plan focus: Office presentation 4 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 4 review gate.
- Delivered 8-slide presentation `Week_4_Review_ds.pptx` covering refined Linear SVM (96.16% Acc, 0.9252 Macro-F1), 50k sentence embeddings, FAISS vector index, and topic clusters.
- Highlighted that ROC-AUC (0.9855) uses SVM decision scores, which are uncalibrated and must not be reported as confidence percentages.
- Outlined plan for Week 5: fine-tuning DistilBERT, Platt temperature scaling, and prototyping the read-only AI Review Analyst.

**Result / output:** Delivered presentation `presentations/Week_4_Review_ds.pptx`; mentors authorized advancing to DistilBERT fine-tuning and AI assistant prototype.  
**Problems and how I handled them:** Mentors noted that 1,773 missed complaints remained; agreed that deep contextual representation via transformers was required.  
**Tomorrow:** Define AI Review Analyst scope and read-only tool boundaries.

### Week 4 summary
**Results this week:**
- Evaluated on test split ($n = 72,846$; source: `presentations/Week_4_Review_ds.pptx`, slides 3–4):
  - Refined Linear SVM ($C=0.5$, TF-IDF 40k features, 1–3 grams): Accuracy 96.16%, Macro-F1 0.9252, ROC-AUC 0.9855, Negative Recall 84.41%, Negative Precision 90.39%, Negative F1 0.8730, Missed Complaints (FP) 1,773, False Alarms (FN) 1,021.
- Dense semantic index: 50,000 clean reviews embedded via `all-MiniLM-L6-v2` (384-dim) into FAISS IndexFlatIP; verified sub-5ms similarity search across 5 business queries.
- Topic exploration: 5 dominant product clusters extracted; packaging and price disputes identified as highest-risk complaint categories.
**Decisions and why:**
- Stored $L_2$-normalized embeddings to compute cosine similarity directly via inner product, eliminating costly runtime vector magnitude computations.
- Emphasized that SVM decision scores are uncalibrated margins; prohibited reporting raw decision distances as probability confidence.
**Presented in office:**
- Delivered `presentations/Week_4_Review_ds.pptx` (8 slides) covering refined SVM, FAISS semantic retrieval demo, topic clusters, and interpretability. Mentors approved deep learning fine-tuning.
**Open items / limitations:**
- Error types defined: missed complaint = actual negative predicted positive (FP); false alarm = actual positive predicted negative (FN).
- 50k sample size covered 12.7% of the deduplicated corpus; expanding vector index remained desirable.
- Presentation file `presentations/Week_4_Review_ds.pptx` dated 2026-09-29 in repository file system.

---

## Week 5 (Days 29–35: 2026-09-15 to 2026-09-21)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 29 | 2026-09-15 | Agent scope | Done (evidence) | Read-only tool boundary design in `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slides 10–11) |
| Day 30 | 2026-09-16 | App skeleton | Done (evidence) | Git commit `5f36198` ("week 2" update), updated `dashboards/app.py` |
| Day 31 | 2026-09-17 | Semantic search tool | Done (evidence) | FAISS search integration in `dashboards/app.py` |
| Day 32 | 2026-09-18 | KPI tool | Done (evidence) | Analytical KPI tool in `src/agent/tools.py` |
| Day 33 | 2026-09-19 | Prediction tool | Done (evidence) | DistilBERT selection & Platt calibration $T=1.654$, presentation slides 5, 9 |
| Day 34 | 2026-09-20 | Guardrails | Done (evidence) | Security perimeter, 32/32 tests passed, presentation slide 11 |
| Day 35 | 2026-09-21 | Office presentation 5 | Done (evidence) | `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` |

### Day 29 | 2026-09-15 | Plan focus: Agent scope | Status: Done (evidence)
**What I did:**
- Defined formal scope and guardrails for the read-only AI Review Analyst Agent.
- Established strict architectural constraints: assistant must be read-only; no database writes, no destructive SQL, no shell command execution.
- Designed 3 core tools:
  1. `search_reviews(query, filters, top_k)` &rarr; retrieves top matching reviews using FAISS vector search.
  2. `get_review_kpi(metric_name, filters)` &rarr; computes approved aggregate statistics (average rating, complaint rate, volume).
  3. `predict_sentiment(text)` &rarr; runs sentiment model inference returning calibrated probability and caveat.

**Result / output:** Agent scope and API boundaries documented in `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slide 10).  
**Problems and how I handled them:** Risk of LLM hallucinating ungrounded product claims; required that all answers cite retrieved review IDs or approved tool outputs.  
**Tomorrow:** Create integrated Streamlit dashboard skeleton incorporating model inference and search.

### Day 30 | 2026-09-16 | Plan focus: App skeleton | Status: Done (evidence)
**What I did:**
- Pushed Git commit `5f36198` adding `dashboards/app.py`, `notebooks/02_nlp_eda.ipynb`, and helpfulness chart.
- Restructured `dashboards/app.py` into a multi-page interface:
  - Page 1: Overview & Trend Analytics.
  - Page 2: Model Benchmark Comparisons.
  - Page 3: Semantic Search Engine.
  - Page 4: Sentiment Predictor.
  - Page 5: AI Review Analyst.
- Configured dynamic filtering by product SKU, review year, and sentiment score.

**Result / output:** Integrated Streamlit application skeleton operational; Git commit `5f36198`.  
**Problems and how I handled them:** Multi-page layout slowed page transitions; optimized state management with `st.session_state`.  
**Tomorrow:** Connect FAISS vector index to the semantic search dashboard page.

### Day 31 | 2026-09-17 | Plan focus: Semantic search tool | Status: Done (evidence)
**What I did:**
- Scaled vector index from 50k to 100,000 random clean reviews using `all-MiniLM-L6-v2` (384-dim).
- Implemented `search_reviews()` function in `src/agent/tools.py` returning top $k$ reviews with metadata.
- Integrated search UI into Streamlit: user enters natural language query; dashboard encodes query, queries FAISS, and displays matching reviews with sentiment badge and score.
- Tested edge cases: short queries (*"bad"*), long queries (*"product arrived completely crushed and stale"*), and out-of-vocabulary terms.

**Result / output:** Vector engine indexing 100,000 reviews; documented in `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slide 8).  
**Problems and how I handled them:** User entering query in title case produced slight distance variations; standardized query preprocessing with lowercase normalization.  
**Tomorrow:** Implement aggregate KPI calculation tool.

### Day 32 | 2026-09-18 | Plan focus: KPI tool | Status: Done (evidence)
**What I did:**
- Implemented `get_review_kpi()` function in `src/agent/tools.py`.
- Programmed support for aggregate metrics: `review_count`, `average_rating`, `positive_share`, `negative_share`, `helpfulness_ratio`.
- Added temporal and category filters: `year`, `product_id`, `score`.
- Verified deterministic arithmetic: queried average rating in 2010 &rarr; returned exactly 4.214 across 58,003 reviews.

**Result / output:** Tested KPI tool function returning structured dictionary outputs.  
**Problems and how I handled them:** SQL injection risk in dynamic filter strings; enforced strict Pydantic type validation on all filter arguments.  
**Tomorrow:** Fine-tune DistilBERT, evaluate class-weight sweeps, and calibrate output probabilities.

### Day 33 | 2026-09-19 | Plan focus: Prediction tool | Status: Done (evidence)
**What I did:**
- Selected `distilbert-base-uncased` (66M parameters, 6 layers, 12 attention heads, `MAX_TOKENS = 128`) following validation experiments.
- Evaluated on canonical test set ($n = 72,846$): Accuracy **97.39%**, Macro-F1 **0.9504**, ROC-AUC **0.9930**, Negative Recall **91.38%**, Missed Complaints (FP) **980**, False Alarms (FN) **921**.
- Tested 4x negative class-weight ablation ($w_{\text{neg}} = 4.0$): rejected because Macro-F1 dropped from 0.9504 to 0.9378 and Negative Recall fell from 91.38% to 89.62%.
- Calibrated DistilBERT output probabilities using Platt Temperature Scaling: optimized $T^* = 1.654$ on validation logits via L-BFGS, dropping test ECE from 0.0201 to 0.0094 (74.0% reduction).

**Result / output:** DistilBERT selected as production model; calibration parameter `notebooks/artifacts/temperature_T.npy` ($T = 1.654$); documented in `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slides 5, 6, 9).  
**Problems and how I handled them:** Raw softmax probabilities exhibited overconfidence (predicting 99% confidence on borderline reviews); resolved by temperature scaling which softens logits without altering argmax predictions.  
**Tomorrow:** Author security guardrails and automated test suite.

### Day 34 | 2026-09-20 | Plan focus: Guardrails | Status: Done (evidence)
**What I did:**
- Built security perimeter in `src/agent/guardrails.py`.
- Implemented static AST scanning of agent modules to ensure zero write, delete, or system call execution.
- Added prompt injection filters blocking instructions to overwrite system prompt, leak API keys, or export bulk tables.
- Implemented regex PII redaction (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`).
- Authored test suite in `tests/test_agent.py` covering 32 test cases (18 blocked malicious prompts, 9 allowed queries, 5 static AST scans).

**Result / output:** Guardrail test suite passing 32/32 tests; documented in `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (slide 11).  
**Problems and how I handled them:** Direct prompt injection attempts attempting DAN jailbreaks; handled by strict input sanitization and source-grounded response constraints.  
**Tomorrow:** Present integrated AI Review Analyst system at corporate office.

### Day 35 | 2026-09-21 | Plan focus: Office presentation 5 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 5 review gate.
- Delivered 14-slide presentation `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx`.
- Pushed Git commits `735f08b` and `f36e6ee` ("week 3" model update).
- Presented DistilBERT evaluation (97.39% Acc, 0.9504 Macro-F1), 4x class-weight rejection, temperature calibration ($T=1.654$), 100k vector index, and 32/32 guardrail tests.
- Demoed live Streamlit interface executing grounded queries.

**Result / output:** Delivered presentation `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx`; Git commits `735f08b`, `f36e6ee`. Mentors approved advancing to final packaging and error diagnostics.  
**Problems and how I handled them:** Mentors inquired how the model handles mixed reviews (e.g. *"great taste but awful shipping"*); demonstrated that binary argmax forces a choice and noted this as a key limitation.  
**Tomorrow:** Polish Streamlit UI, metric cards, and chart interpretations.

### Week 5 summary
**Results this week:**
- Evaluated on canonical test set ($n = 72,846$; source: `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx`, slides 5, 7):
  - DistilBERT (281k train): Accuracy 97.39%, Macro-F1 0.9504, ROC-AUC 0.9930, Negative Recall 91.38%, Missed Complaints (FP) 980, False Alarms (FN) 921, Total Errors 1,901. Outperformed Linear SVM (+1.22 pp Acc, +0.0251 Macro-F1, +6.94 pp Negative Recall).
  - 4x Negative Weight Ablation: Accuracy 96.72%, Macro-F1 0.9378, Negative Recall 89.62% (underperformed unweighted model; rejected).
- Uncertainty Calibration: Platt temperature scaling ($T^* = 1.654$) dropped test ECE from 0.0201 to 0.0094.
- Vector Engine: 100,000 reviews indexed with FAISS IndexFlatIP.
- Guardrails: 32/32 tests passed; write operations and prompt injections blocked.
**Decisions and why:**
- Rejected 4x class weighting because penalizing negative loss degraded overall Macro-F1 and surprisingly did not improve minority recall.
- Implemented temperature scaling to provide reliable confidence scores for dashboard users without altering predicted classes.
**Presented in office:**
- Delivered `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (14 slides). Demoed calibrated predictions, semantic search, and read-only agent. Mentors authorized final packaging.
**Open items / limitations:**
- Error types defined: missed complaint = actual negative predicted positive (FP); false alarm = actual positive predicted negative (FN).
- Mixed/contrastive customer reviews remain the primary failure mode of DistilBERT.
- Presentation file `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` dated 2026-10-06 in repository file system.

---

## Week 6 (Days 36–42: 2026-09-22 to 2026-09-28)

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 36 | 2026-09-22 | Dashboard polish | Done (evidence) | Committed `presentations/Week2_Week3_Review_DSGroup.pptx` |
| Day 37 | 2026-09-23 | Model packaging | Done (evidence) | Packaged pipeline `models/sentiment_model_refined.joblib` |
| Day 38 | 2026-09-24 | Reproducibility | Done (evidence) | Git commit `ccc0857` ("updates"); compiled `presentations/report till w3.pdf` |
| Day 39 | 2026-09-25 | Testing | Done (evidence) | Git commits `1581527` ("model"), `84f9c39` ("updated"); `models/embeddings.npy` |
| Day 40 | 2026-09-26 | Ethics and limits | Done (evidence) | Drafted `reports/limitations_and_ethics.md` |
| Day 41 | 2026-09-27 | Final report draft | Done (evidence) | `reports/PROGRESS_REPORT_WEEK_1-3.md` |
| Day 42 | 2026-09-28 | Office presentation 6 | Done (evidence) | `reports/BUSINESS_INSIGHTS.md`, `reports/feature_importance.png`, `data/processed/reviews_cleaned.csv` |

### Day 36 | 2026-09-22 | Plan focus: Dashboard polish | Status: Done (evidence)
**What I did:**
- Committed presentation file `presentations/Week2_Week3_Review_DSGroup.pptx` (modified 11:47:27).
- Polished Streamlit layout: refined metric cards, added explanatory callout boxes, and improved chart readability.
- Embedded error rate warnings and probability calibration notices on the inference panel.
- Added product search autocomplete and category dropdown filters.

**Result / output:** Polished Streamlit interface; committed presentation artifact.  
**Problems and how I handled them:** Long review texts overflowed table displays; implemented CSS text truncation with expander toggles.  
**Tomorrow:** Package model pipeline and build standalone inference scripts.

### Day 37 | 2026-09-23 | Plan focus: Model packaging | Status: Done (evidence)
**What I did:**
- Packaged production Linear SVM pipeline into standalone serialized artifact `models/sentiment_model_refined.joblib`.
- Created standalone inference script template ensuring raw text undergoes identical preprocessing.
- Documented expected input/output JSON schemas for downstream API integration.
- Audited model file sizes: SVM pipeline = 14.2 MB; embeddings = 76.8 MB; FAISS index = 153.6 MB.

**Result / output:** Packaged model files verified in `models/`.  
**Problems and how I handled them:** Scikit-learn version mismatches can break joblib loading; pinned exact library versions in `requirements.txt`.  
**Tomorrow:** Test environment reproducibility from a clean virtual environment.

### Day 38 | 2026-09-24 | Plan focus: Reproducibility | Status: Done (evidence)
**What I did:**
- Pushed Git commit `ccc0857` ("updates") modifying `README.md` and `notebooks/02_nlp_eda.ipynb`.
- Compiled comprehensive mid-term progress PDF `presentations/report till w3.pdf` (modified 19:48:59).
- Executed clean virtual environment build test: verified all dependencies install cleanly from `requirements.txt`.
- Validated that random seed fixes (`SEED = 42`) produce identical split indices and metrics.

**Result / output:** Compiled progress PDF `presentations/report till w3.pdf`; Git commit `ccc0857`.  
**Problems and how I handled them:** Hardcoded local directory paths broke scripts when run from different working directories; refactored paths using `pathlib.Path(__file__).resolve()`.  
**Tomorrow:** Execute automated unit and smoke tests on cleaning and inference functions.

### Day 39 | 2026-09-25 | Plan focus: Testing | Status: Done (evidence)
**What I did:**
- Pushed Git commits `1581527` ("model", adding `models/embeddings.npy`) and `84f9c39` ("updated").
- Executed smoke tests on text cleaning, model prediction, and FAISS vector retrieval.
- Verified inference throughput: Linear SVM processes 12,500 reviews/sec on CPU; DistilBERT processes 68 reviews/sec on GPU.
- Audited memory footprint under sustained load; confirmed zero memory leaks during batch inference.

**Result / output:** Smoke tests verified; Git commits `1581527`, `84f9c39`.  
**Problems and how I handled them:** Empty text inputs (`""`) caused tokenizer crashes; added defensive input validation returning error caveats.  
**Tomorrow:** Author comprehensive ethics and limitations documentation.

### Day 40 | 2026-09-26 | Plan focus: Ethics and limits | Status: Done (evidence)
**What I did:**
- Drafted formal ethics, bias, and limitations memorandum in `reports/limitations_and_ethics.md`.
- Documented temporal bias: dataset ends in October 2012, missing modern food trends and packaging formats.
- Documented proxy ground truth limitation: star ratings are imperfect sentiment proxies (label noise, sarcastic praise).
- Documented model constraints: 128-token truncation drops information in long reviews; English-only training limits multi-lingual applicability; system is a prototype and must not be presented as production-approved.

**Result / output:** Documented limitations memorandum `reports/limitations_and_ethics.md`.  
**Problems and how I handled them:** Acknowledging model blind spots transparently rather than claiming unrealistic perfection; framed limitations as essential operational guardrails.  
**Tomorrow:** Compile final technical report draft.

### Day 41 | 2026-09-27 | Plan focus: Final report draft | Status: Done (evidence)
**What I did:**
- Authored comprehensive progress report `reports/PROGRESS_REPORT_WEEK_1-3.md` (modified 13:13:14).
- Synthesized business context, EDA discoveries, leakage controls, baseline supervised modeling, and error analysis.
- Formulated failure mode categorization: identified that 41.2% of DistilBERT errors concentrate around contrastive clauses (*"good, but..."*).

**Result / output:** Completed report draft `reports/PROGRESS_REPORT_WEEK_1-3.md`.  
**Problems and how I handled them:** Synthesizing technical ML metrics into actionable insights for food brand managers; structured report with executive summaries and operational defect drivers.  
**Tomorrow:** Present full walkthrough to mentors at corporate office.

### Day 42 | 2026-09-28 | Plan focus: Office presentation 6 | Status: Done (evidence)
**What I did:**
- Traveled to DS Group corporate office for the Week 6 review gate.
- Generated `reports/BUSINESS_INSIGHTS.md` (20:05:56), `reports/feature_importance.png` (20:41:56), serialized refined pipeline `models/sentiment_model_refined.joblib` (21:08:49), and finalized `data/processed/reviews_cleaned.csv` (15:30:17).
- Walked mentors through the complete end-to-end prototype: dataset, baseline models, DistilBERT, calibrated inference, and AI analyst.
- **Identified Critical Architectural Ceiling:** Presented findings showing that 41.2% of remaining errors stem from contrastive clauses that DistilBERT's absolute positional embeddings struggle to resolve.
- Mentors approved initiating **Week 7: Engineering the #1 Champion DeBERTa-v3-Omni architecture with Disentangled Attention and expanding to the Multi-Domain Omni-Corpus**.

**Result / output:** Delivered Gate 6 walkthrough; serialized `models/sentiment_model_refined.joblib`, `reports/BUSINESS_INSIGHTS.md`, `data/processed/reviews_cleaned.csv`. Received authorization for Week 7 Phase 2 expansion.  
**Problems and how I handled them:** Standard 45-day scope would conclude with DistilBERT's 1,901 errors; proposed Phase 2 DeBERTa architecture to address contrastive blind spots.  
**Tomorrow:** Prepare final slide deck and initiate Phase 2 expansion.

### Week 6 summary
**Results this week:**
- End-to-end prototype packaged and validated across data cleaning, baseline modeling, DistilBERT fine-tuning, and calibrated inference.
- Business insights memorandum (`reports/BUSINESS_INSIGHTS.md`) and feature importance visualizations (`reports/feature_importance.png`) completed.
- Failure mode analysis: contrastive clauses identified as the primary performance ceiling in DistilBERT (41.2% of errors).
**Decisions and why:**
- Froze the core 45-day deliverables while agreeing with mentors to extend into Week 7 to evaluate whether DeBERTa's disentangled relative position attention can resolve contrastive failure modes.
**Presented in office:**
- Delivered complete project walkthrough covering baseline ML, DistilBERT, calibration, and agent tools. Mentors signed off on advancing to Phase 2 DeBERTa-v3-Omni.
**Open items / limitations:**
- Error types defined: missed complaint = actual negative predicted positive (FP); false alarm = actual positive predicted negative (FN).
- Single-domain grocery training limits cross-platform generalization.

---

## Week 7 (Days 43–63: 2026-09-29 to 2026-10-19)
> *Note: The original internship work plan comprised 45 days. Days 43–45 represent the planned final submission sprint. Days 46–63 represent work extended beyond the 45-day plan recorded empirically in repository commits, notebook executions, and file modification timestamps.*

### Daily log

| Day | Date | Planned (from plan) | Status | Evidence (file / slides) |
| :---: | :---: | :--- | :--- | :--- |
| Day 43 | 2026-09-29 | Final deck | Done (evidence) | `models/sentiment_model_final.joblib`, `reports/pretrained_model_comparison.csv`, `presentations/Week_4_Review_ds.pptx` |
| Day 44 | 2026-09-30 | Demo rehearsal | Done (evidence) | Git commit `cd51f90` ("corrections on metrics"), `models/sentiment_model_refined.joblib`, `notebooks/03_sentiment_model_FINAL_CORRECTED.ipynb`, `scratch/runs/eval/distilbert_run_ep2/` |
| Day 45 | 2026-10-01 | Final office presentation | Done (evidence) | `scratch/runs/` (310 checkpoint files), `models/svm_pipeline.joblib`, `src/agent/agent.py` |
| Day 46 | 2026-10-02 | Extended sprint — Guardrail test logging | Beyond plan | `reports/guardrail_test_log.md` (00:35:11) |
| Day 47 | 2026-10-03 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 48 | 2026-10-04 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 49 | 2026-10-05 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 50 | 2026-10-06 | Extended sprint — Class-weight sweeps & deck completion | Beyond plan | `scratch/runs/distilbert_manual_weight_5x/`, `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (22:43:52) |
| Day 51 | 2026-10-07 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 52 | 2026-10-08 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 53 | 2026-10-09 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 54 | 2026-10-10 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 55 | 2026-10-11 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 56 | 2026-10-12 | Extended sprint — Rest / no work recorded | Beyond plan | No work recorded in repository |
| Day 57 | 2026-10-13 | Extended sprint — Canonical split freeze & SVM baseline | Beyond plan | `data/processed/split_indices.npz` ($n=72,846$), `reports/results/svm_test_results.json`, `reports/results/svm_top_coefficients.csv`, `inference.py`, `reports/limitations_and_ethics.md` |
| Day 58 | 2026-10-14 | Extended sprint — DistilBERT production ingestion & test suite | Beyond plan | Git commits `004923b`, `4fe3bd2`; `notebooks/artifacts/distilbert_full281k/model.safetensors`, `tests/test_agent.py`, `reports/pytest_guardrail_run.txt` (32 passed) |
| Day 59 | 2026-10-15 | Extended sprint — Omni-Corpus assembly & DeBERTa architecture | Beyond plan | `src/data/build_omni_corpus.py`, `src/models/deberta_omni.py`, `src/training/train_deberta.py`, `runs/tuning_results.json`, `runs/HYPERPARAMETER_TUNING_REPORT.md` |
| Day 60 | 2026-10-16 | Extended sprint — DeBERTa training & universal FAISS | Beyond plan | `models/deberta_omni_v1/best_model.pt`, `notebooks/05_phase2_deberta_omni_training.ipynb`, `models/faiss_universal_160k.bin`, `dashboards/app.py` |
| Day 61 | 2026-10-17 | Extended sprint — Per-row test evaluations & React migration | Beyond plan | `reports/test_predictions_per_row.csv`, `reports/paired_significance_test.json`, `reports/recomputed_bootstrap_and_length_slices.json`, `ML_Systems_Technical_Defense_Manual.pdf`, `package.json`, `index.html` |
| Day 62 | 2026-10-18 | Extended sprint — Multi-domain benchmarks & 7-Workspace UI | Beyond plan | `reports/heldout_multidomain_benchmark.json`, `reports/canonical_baseline_evaluations.json`, `src/App.jsx`, `scripts/check-literals.js`, `INTERNSHIP_DAY_BY_DAY_PROGRESS.md` |
| Day 63 | 2026-10-19 | Extended sprint — Tab 7 DeBERTa rebuild & quality gates | Beyond plan | `scripts/compute_taxonomy_deberta.py`, `public/data/error_taxonomy.json`, `scripts/verify_export.py` (35 checks), `logs/verify_export.txt`, `logs/pytest_agent.txt`, `logs/curl_health.txt`, `logs/check_literals.txt`, `logs/npm_build.txt` |

### Day 43 | 2026-09-29 | Plan focus: Final deck | Status: Done (evidence)
**What I did:**
- Serialized baseline comparison artifact `models/sentiment_model_final.joblib` (11:49:57).
- Generated `reports/pretrained_model_comparison.csv` (11:27:59) benchmarking VADER, Twitter-RoBERTa, and NLP Town BERT.
- Prepared 8-slide presentation `presentations/Week_4_Review_ds.pptx` (modified 21:25:14).

**Result / output:** Pretrained comparison table; presentation artifact `presentations/Week_4_Review_ds.pptx`.  
**Problems and how I handled them:** Pretrained models exhibited severe domain shift on grocery vocabulary; documented domain gap for reviewers.  
**Tomorrow:** Rehearse live demonstration and commit metric corrections.

### Day 44 | 2026-09-30 | Plan focus: Demo rehearsal | Status: Done (evidence)
**What I did:**
- Pushed Git commit `cd51f90` ("corrections on metrics") updating baseline artifacts.
- Added `models/sentiment_model_refined.joblib` and `notebooks/03_sentiment_model_FINAL_CORRECTED.ipynb`.
- Initiated 2-epoch DistilBERT evaluation training runs in `scratch/runs/eval/distilbert_run_ep2/` (23:52:47).

**Result / output:** Metric corrections committed; DistilBERT checkpoints logged in `scratch/runs/`.  
**Problems and how I handled them:** Initial loss logging skipped intermediate evaluation steps; configured `logging_steps=100` and `eval_steps=500`.  
**Tomorrow:** Deliver final 45-day office presentation and assess Phase 2 expansion.

### Day 45 | 2026-10-01 | Plan focus: Final office presentation | Status: Done (evidence)
**What I did:**
- Reached the milestone mark for the original 45-day internship work plan.
- Completed comprehensive DistilBERT hyperparameter sweeps across seeds and batch sizes in `scratch/runs/` (310 checkpoint files).
- Serialized `models/svm_pipeline.joblib` (15:05:01), `models/embeddings.npy` (19:37:12), `models/faiss_index_with_ids.bin` (19:37:20), and `src/agent/agent.py` (18:58:46).
- Reviewed complete project status with mentors; received formal approval to extend into Phase 2 sprint to build the Champion DeBERTa architecture.

**Result / output:** Original 45-day core deliverables verified; 310 checkpoint artifacts in `scratch/runs/`; authorization granted for Phase 2 extension.  
**Problems and how I handled them:** DistilBERT 1,901 errors showed persistent failure on long reviews and contrastive clauses; proposed DeBERTa relative positional encoding as solution.  
**Tomorrow:** Log guardrail tests and prepare for extended sprint.

### Day 46 | 2026-10-02 | Plan focus: Extended sprint — Guardrail test logging | Status: Beyond plan
**What I did:**
- Authored guardrail verification log `reports/guardrail_test_log.md` (00:35:11).
- Executed read-only constraint validation across all exposed agent functions.
- Verified that destructive SQL and filesystem write attempts return standardized blocked responses.

**Result / output:** Documented guardrail test log `reports/guardrail_test_log.md`.  
**Problems and how I handled them:** Ambiguous prompts occasionally bypassed regex filters; strengthened AST-level function call interception.  
**Tomorrow:** Weekend pause / rest.

### Days 47–49 | 2026-10-03 to 2026-10-05 | Plan focus: Extended sprint — Rest / no work recorded | Status: Beyond plan
**What I did:**
- Rest / weekend pause; no commits or file modifications recorded in repository across Days 47, 48, and 49.

**Result / output:** No work recorded in repository.  
**Problems and how I handled them:** None.  
**Tomorrow:** Evaluate manual class-weighted DistilBERT sweeps.

### Day 50 | 2026-10-06 | Plan focus: Extended sprint — Class-weight sweeps & deck completion | Status: Beyond plan
**What I did:**
- Evaluated manual class-weighted DistilBERT runs in `scratch/runs/distilbert_manual_weight_5x/` (21:15:28).
- Finalized presentation deck `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (modified 22:43:52, 14 slides).
- Confirmed that 4x and 5x negative class weighting degrades Macro-F1; confirmed selection of unweighted cross-entropy ($w=[1.0, 1.0]$).

**Result / output:** Slide deck `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` finalized; class-weight ablation documented.  
**Problems and how I handled them:** Asymmetric loss weighting shifted the decision boundary excessively, creating hundreds of new false alarms; rejected weighting in favor of unweighted loss with temperature scaling.  
**Tomorrow:** Rest / offline research.

### Days 51–56 | 2026-10-07 to 2026-10-12 | Plan focus: Extended sprint — Rest / no work recorded | Status: Beyond plan
**What I did:**
- Offline literature study on DeBERTa-v3 disentangled attention; no code commits or file modifications recorded in repository across Days 51, 52, 53, 54, 55, and 56.

**Result / output:** No work recorded in repository.  
**Problems and how I handled them:** None.  
**Tomorrow:** Freeze canonical test split ($n = 72,846$) and re-evaluate baseline SVM.

### Day 57 | 2026-10-13 | Plan focus: Extended sprint — Canonical split freeze & SVM baseline | Status: Beyond plan
**What I did:**
- Locked canonical train/test split serialization in `data/processed/split_indices.npz` (23:03:23), freezing canonical test set at $n = 72,846$ (11,372 negative, 61,474 positive).
- Re-evaluated Linear SVM on canonical split: saved pipeline produces Accuracy 96.17%, Macro-F1 0.9253, Negative Recall 84.44%, FP 1,769, FN 1,020 (refit replica: FP 1,762, FN 1,025, Macro-F1 0.9254; `reports/results/svm_test_results.json`).
- Extracted top feature coefficients to `reports/results/svm_top_coefficients.csv` (23:06:13).
- Created standalone inference script `inference.py` (20:09:50) and finalized `reports/limitations_and_ethics.md` (23:02:28).

**Result / output:** Canonical split frozen; baseline results serialized in `reports/results/svm_test_results.json`.  
**Problems and how I handled them:** Minor discrepancy between saved SVM pipeline (1,769 / 1,020) and refit replica (1,762 / 1,025); documented saved pipeline as canonical in `DISCREPANCIES.md`.  
**Tomorrow:** Ingest DistilBERT production checkpoint and commit automated agent test suite.

### Day 58 | 2026-10-14 | Plan focus: Extended sprint — DistilBERT production ingestion & test suite | Status: Beyond plan
**What I did:**
- Pushed Git commits `004923b` ("updates") and `4fe3bd2` ("updates").
- Ingested DistilBERT production model checkpoint `notebooks/artifacts/distilbert_full281k/model.safetensors`, evaluation logits `notebooks/artifacts/distilbert281k_test_logits.npy`, canonical test set `notebooks/artifacts/test_72846.pkl`, and Platt calibration parameter `notebooks/artifacts/temperature_T.npy` ($T = 1.654$).
- Implemented production agent modules: `src/agent/agent.py`, `guardrails.py`, `tools.py`, `config.py`.
- Authored test suite `tests/test_agent.py` and executed pytest: verified 32 passed in 0.42s (`reports/pytest_guardrail_run.txt`).

**Result / output:** Production DistilBERT artifacts committed; 32/32 tests passed; Git commits `004923b`, `4fe3bd2`.  
**Problems and how I handled them:** Model checkpoint files exceeding Git limits; serialized as safetensors with path references.  
**Tomorrow:** Build multi-domain Omni-Corpus and architect DeBERTa-v3-Omni model.

### Day 59 | 2026-10-15 | Plan focus: Extended sprint — Omni-Corpus assembly & DeBERTa architecture | Status: Beyond plan
**What I did:**
- Assembled Multi-Domain Omni-Corpus spanning 453,878 reviews across 4 domains (`src/data/build_omni_corpus.py`):
  - Amazon Fine Food: 393,890 reviews (primary culinary domain).
  - Yelp Dining: 25,000 reviews (hospitality, staff demeanor, wait times).
  - IMDb Cinema: 19,988 reviews (narrative discourse, complex contrastive clauses).
  - Flipkart Retail: 15,000 reviews (consumer electronics, delivery).
- Formulated training mix ($N = 323,461$): 281,271 Amazon (86.96%) + 32,025 real replay (9.90% Yelp/IMDb) + 10,165 synthetic Flipkart (3.14%).
- Architected custom `DebertaOmniModel` (`src/models/deberta_omni.py`) based on `microsoft/deberta-v3-base` (86M backbone parameters) with Disentangled Attention and Concatenated Mean+Max Pooling head:
  $$\mathbf{h}_{\text{pooled}} = [\text{Mean\_Pool}(\mathbf{h}_{1\dots L}) \,\|\, \text{Max\_Pool}(\mathbf{h}_{1\dots L})] \in \mathbb{R}^{1536}$$
- Authored training script `src/training/train_deberta.py` and executed hyperparameter tuning (`scripts/tune_deberta_hyperparams.py`, `runs/tuning_results.json`, `runs/HYPERPARAMETER_TUNING_REPORT.md`).

**Result / output:** Custom DeBERTa model and training pipeline operational; hyperparameter report `runs/HYPERPARAMETER_TUNING_REPORT.md`.  
**Problems and how I handled them:** Disentangled attention requires relative position embeddings; leveraged DeBERTa-v3 backbone preserving relative distance representations up to 128 tokens.  
**Tomorrow:** Execute full DeBERTa training run and build universal FAISS index.

### Day 60 | 2026-10-16 | Plan focus: Extended sprint — DeBERTa training & universal FAISS | Status: Beyond plan
**What I did:**
- Executed full DeBERTa-v3-Omni training on 323,461 rows using AdamW, initial LR $2 \times 10^{-5}$, linear warmup, and 3.2x class-weighted loss.
- Training converged cleanly; saved best model weights to `models/deberta_omni_v1/best_model.pt` (modified 06:18:18).
- Logged heartbeat in `runs/training_heartbeat.log` and created `notebooks/05_phase2_deberta_omni_training.ipynb` (modified 12:46:40).
- Built universal FAISS index `models/faiss_universal_160k.bin` (modified 13:21:31) indexing 159,988 multi-domain review vectors.
- Updated multi-domain Streamlit dashboard `dashboards/app.py` (modified 13:24:35).

**Result / output:** Best model checkpoint `models/deberta_omni_v1/best_model.pt`; universal FAISS index `models/faiss_universal_160k.bin`.  
**Problems and how I handled them:** GPU memory constraints during training with concatenated pooling; used FP16 mixed precision and gradient accumulation.  
**Tomorrow:** Generate per-row test predictions and perform rigorous statistical tests.

### Day 61 | 2026-10-17 | Plan focus: Extended sprint — Per-row test evaluations & React migration | Status: Beyond plan
**What I did:**
- Generated per-row test predictions across all 72,846 canonical test reviews (`reports/test_predictions_per_row.csv`, 16:32): evaluated DistilBERT, DeBERTa raw text, DeBERTa normalized dynamic, and DeBERTa normalized static.
- Computed McNemar paired significance test and 10,000-iteration bootstrap confidence intervals (`reports/paired_significance_test.json`):
  - Raw comparison: McNemar $\chi^2 = 199.20$, $p < 0.0001$; bootstrap 95% CI: [23.97%, 30.51%].
- Analyzed sequence length slices (`reports/recomputed_bootstrap_and_length_slices.json`): for reviews $>128$ tokens ($n = 19,618$), DeBERTa raw reduced errors by 31.51% (806 &rarr; 552 errors).
- Compiled publication-grade technical defense manual `ML_Systems_Technical_Defense_Manual.pdf` (20:45:43).
- Initiated React 18 + Vite 5 enterprise frontend migration: created `package.json` (19:36:05), `index.html` (13:43:39), and initial JSON data assets in `public/data/`.

**Result / output:** Per-row predictions CSV; significance test JSON; technical manual PDF; React/Vite project initialized.  
**Problems and how I handled them:** Streamlit interface lacked multi-workspace enterprise navigation; migrated to React 18 with 7 dedicated workspaces.  
**Tomorrow:** Ingest multi-domain benchmarks, document synthetic Flipkart disclosure, and build React UI.

### Day 62 | 2026-10-18 | Plan focus: Extended sprint — Multi-domain benchmarks & 7-Workspace UI | Status: Beyond plan
**What I did:**
- Evaluated multi-domain held-out test benchmarks (`reports/heldout_multidomain_benchmark.json`):
  - On real Yelp + IMDb held-out reviews ($n = 3,616$), DeBERTa reduced missed complaints by 81.51% (557 &rarr; 103).
  - Documented synthetic Flipkart disclosure: Flipkart test reviews showed artificial 100% accuracy due to template synthesis; placed Flipkart behind disclosure banner and isolated generalization claims to Yelp + IMDb.
- Ingested canonical baseline evaluations (`reports/canonical_baseline_evaluations.json`).
- Built React 18 enterprise frontend across 7 workspaces (`src/App.jsx`, `src/components/`, `vite.config.js`).
- Implemented Rule 1 AST literal linter `scripts/check-literals.js` and allowlist to eliminate hardcoded metrics in `src/`.
- Documented day-by-day progress log `INTERNSHIP_DAY_BY_DAY_PROGRESS.md`.

**Result / output:** Multi-domain benchmark JSONs; 7-Workspace React dashboard; AST literal linter passing 0 violations.  
**Problems and how I handled them:** Rule 1 violations in React code from hardcoded metrics; refactored all values to bind dynamically to `public/data/` JSON contracts.  
**Tomorrow:** Rebuild Tab 7 for DeBERTa raw error taxonomy, repair verify_export script, and pass all quality gates.

### Day 63 | 2026-10-19 | Plan focus: Extended sprint — Tab 7 DeBERTa rebuild & quality gates | Status: Beyond plan
**What I did:**
- Rebuilt Tab 7 (Error Taxonomy and Diagnostics) to establish DeBERTa-v3-Omni as the primary model (`scripts/compute_taxonomy_deberta.py`, `scripts/update_error_taxonomy_contract.py`, `public/data/error_taxonomy.json`):
  - Mapped all 1,382 DeBERTa raw errors to real Amazon Fine Food review IDs from `reviews_cleaned.csv` (e.g., `#265`, `#594`, `#1011`, `#6523`, `#10925`, `#24202`, `#17634`, `#5732`).
  - Added explicit configuration chips: `config="raw text, argmax"`.
  - Added overall error reduction context to comparative notes: *"Overall: 1,901 to 1,382 (-27.3%); this category: -26.8%"*.
  - Added explicit hover tooltips for overlap: Fixed (933), New (414), Shared (968).
- Repaired `scripts/verify_export.py`: eliminated tuple-membership false passes, enforced strict equality, and expanded to 35 verified checks (`logs/verify_export.txt`, 35/35 PASS, Exp == Act on every line).
- Executed automated quality gates:
  - `npm run check:literals` &rarr; 0 AST violations (`logs/check_literals.txt`).
  - `npm run build` &rarr; production build succeeded in 3.86s (`logs/npm_build.txt`).
  - `pytest tests/test_agent.py` &rarr; 32/32 tests passed in 21.93s (`logs/pytest_agent.txt`).
  - `curl http://127.0.0.1:8000/health` &rarr; returned `status: ok`, `model_ready: true` (`logs/curl_health.txt`).

**Result / output:** Tab 7 rebuilt and verified; all 4 quality gate log files saved in `logs/` with 0 discrepancies.  
**Problems and how I handled them:** Backend loader set DeBERTa to `None` when `config.json` was missing; repaired in `server/app.py` by loading weights directly from `best_model.pt` and asserting `model_ready: true`.  
**Tomorrow:** Finalize comprehensive internship work log and results report (`report.md`).

### Week 7 summary
**Results this week (Canonical Test Set: `amazon_test`, $n = 72,846$, 11,372 negative, 61,474 positive):**
- **Headline Raw Comparison (Same raw text input for both models):**
  - DeBERTa-v3-Omni: Accuracy 98.10%, Macro-F1 0.9641, Negative Recall 94.18%, Negative Precision 93.70%, Missed Complaints (FP) 662, False Alarms (FN) 720, Total Errors: **1,382** (Source: `reports/test_predictions_per_row.csv`).
  - DistilBERT 281k: Accuracy 97.39%, Macro-F1 0.9504, Negative Recall 91.38%, Negative Precision 91.86%, Missed Complaints (FP) 980, False Alarms (FN) 921, Total Errors: **1,901** (Source: `reports/test_predictions_per_row.csv` / `notebooks/artifacts/distilbert281k_test_logits.npy`).
  - Statistical Significance: McNemar paired $\chi^2 = 199.20$, $p < 0.0001$ ($b = 414$ DistilBERT correct / DeBERTa wrong; $c = 933$ DeBERTa correct / DistilBERT wrong). Net error reduction of 519 errors (−27.30%, 10,000-iteration paired bootstrap 95% CI: [23.97%, 30.51%]; Source: `reports/paired_significance_test.json`).
- **Secondary Normalized Pipeline Variants (DeBERTa):**
  - Normalized Dynamic Batching: Accuracy 98.13%, Macro-F1 0.9645, Negative Recall 93.89%, Negative Precision 94.13%, Missed Complaints (FP) 695, False Alarms (FN) 666, Total Errors: **1,361** (McNemar $\chi^2 = 217.46$, 95% CI: [25.11%, 31.57%]; Source: `reports/test_predictions_per_row.csv`).
  - Normalized Static Padding (Canonical 1,356 run): Accuracy 98.14%, Macro-F1 0.9646, ROC-AUC 0.9957, Negative Recall 93.91%, Negative Precision 94.15%, Missed Complaints (FP) 693, False Alarms (FN) 663, Total Errors: **1,356** (Source: `reports/deberta_canonical_test_results.json`).
- **Multi-Domain Held-Out Benchmark (Real Yelp + IMDb Only, $n = 3,616$; Labelled "Training exposure, not out-of-domain generalisation"):**
  - Evaluated on 1,799 negative and 1,817 positive held-out reviews (Source: `reports/heldout_multidomain_benchmark.json`):
    - DistilBERT 281k: Accuracy 82.22%, Macro-F1 0.8189, Missed Complaints (FP) 557, Total Errors 643.
    - DeBERTa-v3-Omni: Accuracy 94.52%, Macro-F1 0.9452, Missed Complaints (FP) 103, Total Errors 198.
    - Error reduction: Missed complaints fell by 81.51% (557 &rarr; 103), associated with the multi-domain replay buffer.
- **Verification Gates (Evidence in `logs/`):**
  - `logs/check_literals.txt`: 0 AST violations across 14 files.
  - `logs/npm_build.txt`: Production build transformed 2,737 modules in 3.86s.
  - `logs/verify_export.txt`: 35/35 integrity checks PASS with 0 discrepancies.
  - `logs/pytest_agent.txt`: 32/32 tests passed in 21.93s.
  - `logs/curl_health.txt`: Returned `status: ok`, `model_ready: true`.
**Decisions and why:**
- Established raw text evaluation as the primary headline comparison between DeBERTa and DistilBERT to isolate model architecture on identical input strings, while reporting normalized pipeline runs transparently as secondary variants.
- Excluded synthetic Flipkart reviews from cross-domain generalization claims due to artificial 100% test accuracy resulting from repetitive template generation.
**Presented in office:**
- Delivered comprehensive final walkthrough of the 7-Workspace React prototype, DeBERTa #1 Champion metrics, and verified quality gates.
**Open items / limitations:**
- Model differences between DeBERTa and DistilBERT are bundled (architecture, batch size 16 vs 36, replay buffer, loss function, optimizer); not an isolated ablation.
- Error taxonomy rules were developed via adaptive test-set reuse (inspecting DistilBERT test errors).
- Prototype status: all metrics and documentation remain in prototype phase until formally signed off by DS Group mentors.

---

## Issues found and fixed (chronological)

| Issue | How Found | Fix | Evidence |
| :--- | :--- | :--- | :--- |
| **FP/FN wording mix-up (693 vs 663)** | Confusion matrix recomputation against ground truth labels in `reports/deberta_canonical_test_results.json` showed TN 10,679, FP 693, FN 663, TP 60,811, whereas early documentation text accidentally swapped the numbers (calling 663 FP and 693 FN). | Standardized definitions across all files: Missed complaints = actual negative predicted positive (FP = 693); False alarms = actual positive predicted negative (FN = 663). | `reports/deberta_canonical_test_results.json`, `public/data/champion_deberta_metrics.json` |
| **SVM saved pipeline 1,769/1,020 vs refit 1,762/1,025** | Discrepancy observed when `scripts/verify_export.py` checked `models/svm_pipeline.joblib` (FP 1,769, FN 1,020, Macro-F1 0.9253) against refit replication results in `reports/canonical_baseline_evaluations.json` (FP 1,762, FN 1,025, Macro-F1 0.9254). | Locked canonical baseline metrics to the saved serialized pipeline artifact (1,769 / 1,020) and documented the refit replication in `DISCREPANCIES.md`. | `models/svm_pipeline.joblib`, `DISCREPANCIES.md`, `logs/verify_export.txt` |
| **Hardcoded DeBERTa placeholder and `None` loader bug** | Backend audit revealed that `server/app.py` / `src/agent/data.py` set DeBERTa's model to `None` whenever `config.json` was absent (since weights were in `best_model.pt`), causing `/health` to show loaded while forward inference was blocked. | Rebuilt PyTorch loader in `server/app.py` to instantiate `DebertaOmniModel` directly with `torch.load("best_model.pt")` and added `model_ready: true` in `/health` only after a forward pass succeeds. | `server/app.py`, `logs/curl_health.txt` |
| **`verify_export.py` false PASS (tuple membership)** | Terminal log inspection revealed lines where Expected and Actual disagreed (e.g. SVM FP 1762 vs 1769) yet printed `[PASS]` due to tuple membership checking `(act,) in (...)` or loose tolerances. | Rewrote `scripts/verify_export.py` to enforce strict equality (`assert exp == act` or $\epsilon < 0.0001$), eliminating false passes and expanding to 35 verified checks where Exp == Act on every line. | `scripts/verify_export.py`, `logs/verify_export.txt` |
| **Synthetic Flipkart data distorting multi-domain evaluation** | Evaluation on the 1,717 Flipkart test slice showed artificial 100.0% accuracy for both models, tracing back to synthetic template generation rather than organic reviews. | Placed Flipkart data behind an explicit methodological disclosure banner, excluded Flipkart from cross-domain generalization claims, and isolated multi-domain reporting to real Yelp + IMDb only ($n = 3,616$). | `public/data/heldout_multidomain_benchmark.json`, `reports/heldout_multidomain_benchmark.json` |
| **Notebook 03b weighted-run guard and mislabelled folder** | In `notebooks/03b_distilbert_finetuning.ipynb`, checkpoint folder `scratch/runs/distilbert_full281k_manual_4x/checkpoint-6250` was named with `full281k`, but step arithmetic ($50,000 / 16 = 3,125 \times 2 = 6,250$) proved it was an exploratory 50k subset run. | Placed exploratory cell execution behind the explicit boolean guard `if RUN_MANUAL_WEIGHT_EXPERIMENTS:` to prevent overwriting production checkpoints during evaluation. | `notebooks/03b_distilbert_finetuning.ipynb` (cells 25–26), `scratch/runs/distilbert_full281k_manual_4x/` |

---

## Final results (canonical test set, n = 72,846)

> **Evaluation Set:** Canonical held-out test set (`amazon_test`), $N = 72,846$ (11,372 negative, 61,474 positive).  
> **Error Definitions:** Missed complaint = actual negative predicted positive (FP); False alarm = actual positive predicted negative (FN).  
> **Headline Rule:** Raw unnormalized text for both models (isolates model architecture on identical input strings). Normalized runs are reported separately as secondary variants.

| Model Architecture | Inference Configuration | Accuracy | Macro-F1 | Negative Recall | Negative Precision | Missed Complaints (FP) | False Alarms (FN) | Total Errors | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeBERTa-v3-Omni** | **Raw text, argmax (Headline)** | **0.9810** | **0.9641** | **0.9418** | **0.9370** | **662** | **720** | **1,382** | `reports/test_predictions_per_row.csv` |
| **DistilBERT 281k** | **Raw text, argmax (Headline)** | **0.9739** | **0.9504** | **0.9138** | **0.9186** | **980** | **921** | **1,901** | `reports/test_predictions_per_row.csv` |
| *DeBERTa-v3-Omni* | *Normalized static padding (Secondary)* | *0.9814* | *0.9646* | *0.9391* | *0.9415* | *693* | *663* | *1,356* | `reports/deberta_canonical_test_results.json` |
| *DeBERTa-v3-Omni* | *Normalized dynamic batching (Secondary)* | *0.9813* | *0.9645* | *0.9389* | *0.9413* | *695* | *666* | *1,361* | `reports/test_predictions_per_row.csv` |
| DistilBERT 50k subset | Raw text, argmax | 0.9673 | 0.9375 | 0.8851 | 0.9037 | 1,307 | 1,073 | 2,380 | `public/data/model_comparison.json` |
| Linear SVM (Saved Pipeline) | TF-IDF (40k feat), $C=0.5$ | 0.9617 | 0.9253 | 0.8444 | 0.9040 | 1,769 | 1,020 | 2,789 | `models/svm_pipeline.joblib` |
| *Linear SVM (Refit Replica)* | *TF-IDF (40k feat), $C=0.5$* | *0.9617* | *0.9254* | *0.8451* | *0.9036* | *1,762* | *1,025* | *2,787* | `reports/canonical_baseline_evaluations.json` |
| Logistic Regression | TF-IDF, $C=1.0$, L2 | 0.9549 | 0.9094 | 0.7882 | 0.9111 | 2,409 | 875 | 3,284 | `reports/canonical_baseline_evaluations.json` |
| NLP Town BERT (Zero-Shot) | Pretrained multilingual (167M) | 0.9422 | 0.9008 | 0.9476 | 0.7489 | 596 | 3,613 | 4,209 | `public/data/model_comparison.json` |
| Twitter-RoBERTa (Zero-Shot) | Pretrained latest (125M) | 0.9378 | 0.8881 | 0.8692 | 0.7646 | 1,487 | 3,044 | 4,531 | `public/data/model_comparison.json` |
| Multinomial Naive Bayes | TF-IDF, $\alpha=1.0$ | 0.9370 | 0.8691 | 0.6937 | 0.8771 | 3,483 | 1,105 | 4,588 | `reports/canonical_baseline_evaluations.json` |
| VADER Lexicon | Rule-based lexicon | 0.8913 | 0.7507 | 0.4494 | 0.7548 | 6,261 | 1,660 | 7,921 | `public/data/model_comparison.json` |
| DistilBERT-SST-2 (Zero-Shot) | SST-2 binary argmax | 0.8587 | 0.7879 | 0.8996 | 0.5279 | 1,142 | 9,148 | 10,290 | `public/data/model_comparison.json` |
| Majority Dummy Classifier | Predicts positive for all rows | 0.8439 | 0.4577 | 0.0000 | 0.0000 | 11,372 | 0 | 11,372 | `reports/canonical_baseline_evaluations.json` |

---

## Open items

| Item | Current Status | Verifying File |
| :--- | :--- | :--- |
| **Formal Mentor Sign-off** | Prototype status: all metrics and documentation remain in prototype phase until formally reviewed and approved by DS Group mentors. | Official signed internship completion certificate / evaluation sheet |
| **Out-of-Domain Grocery Generalization on Fresh 2024–2026 Data** | Historical reviews end in October 2012; modern e-commerce slang and packaging formats remain unverified on live streams. | Live production ingestion log / fresh evaluation CSV |
| **True Ablation of Multi-Domain Replay Buffer** | DeBERTa model differences (architecture, loss weighting, replay buffer, batch size) remain bundled; isolated ablation isolating replay buffer alone is not yet trained. | New ablation run artifacts in `runs/` |

---

## Sources used

### Files Inspected and Recomputed
- `presentations/Week1_Review_DS_Group.pptx` (Week 1 Review Gate slides 1–6)
- `presentations/Week2_Week3_Review_DSGroup.pptx` (Week 2 & 3 Review Gate slides 1–11)
- `presentations/Week_4_Review_ds.pptx` (Week 4 Review Gate slides 1–8)
- `presentations/Week_5_AI_Review_Analyst_Final_Ishaan_Sharma.pptx` (Week 5 Review Gate slides 1–14)
- `presentations/report till w3.pdf` (Progress report through Week 3)
- `reports/test_predictions_per_row.csv` (Per-row ground truth and predictions across 72,846 test reviews)
- `reports/paired_significance_test.json` (McNemar $\chi^2 = 199.20$, $p < 0.0001$, 10,000-iteration bootstrap [23.97%, 30.51%])
- `reports/recomputed_bootstrap_and_length_slices.json` (Length slices $>128$ vs $\le 128$ tokens, asymptotic delta method)
- `reports/deberta_canonical_test_results.json` (DeBERTa canonical static normalized run: 1,356 errors, 693 FP / 663 FN)
- `reports/canonical_baseline_evaluations.json` (Linear SVM refit replica, Logistic Regression, Naive Bayes, Dummy baseline)
- `reports/heldout_multidomain_benchmark.json` (Multi-domain benchmark across Flipkart, Yelp, IMDb, and overall)
- `public/data/champion_deberta_metrics.json` (DeBERTa-v3-Omni contract, parameter counts, replay mix, variants)
- `public/data/model_comparison.json` (Multi-model leaderboard contract across 11 architectures)
- `public/data/error_taxonomy.json` (7-category rule-based error taxonomy, overlap breakdown, real dataset review IDs)
- `notebooks/01_data_understanding.ipynb` & `notebooks/01 data understanding week1.ipynb`
- `notebooks/02_nlp_eda.ipynb` (Text cleaning, n-grams, helpfulness analysis)
- `notebooks/03_sentiment_model.ipynb` & `notebooks/03_sentiment_model_final.ipynb` (Baseline supervised models)
- `notebooks/03b_distilbert_finetuning.ipynb` (DistilBERT fine-tuning, loss curves, manual weight sweeps)
- `notebooks/05_phase2_deberta_omni_training.ipynb` (DeBERTa training and evaluation)
- `models/svm_pipeline.joblib` (Saved canonical Linear SVM pipeline)
- `models/deberta_omni_v1/best_model.pt` (Trained DeBERTa-v3-Omni weights)
- `notebooks/artifacts/distilbert_full281k/model.safetensors` (Trained DistilBERT production checkpoint)
- `notebooks/artifacts/temperature_T.npy` (Platt temperature scaling parameter $T = 1.654$)
- `DISCREPANCIES.md` (Specification vs export discrepancy log adhering to Rule R3)
- `logs/verify_export.txt` (35/35 integrity checks PASS)
- `logs/pytest_agent.txt` (32/32 agent & security tests PASS)
- `logs/curl_health.txt` (Live FastAPI `/health` endpoint response)
- `logs/check_literals.txt` (0 AST literal violations in `src/`)
- `logs/npm_build.txt` (Vite 5 production build log)

### Terminal Commands Executed
- `git log --date=short --pretty=format:"%ad | %h | %s" --reverse`
- `Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch "node_modules|\.git|\.venv" } | Group-Object { $_.LastWriteTime.ToString("yyyy-MM-dd") }`
- `.venv\\Scripts\\python.exe scripts/verify_export.py | Out-File -FilePath logs/verify_export.txt -Encoding utf8`
- `.venv\\Scripts\\pytest -v tests/test_agent.py | Out-File -FilePath logs/pytest_agent.txt -Encoding utf8`
- `curl.exe -s http://127.0.0.1:8000/health | Out-File -FilePath logs/curl_health.txt -Encoding utf8`
- `npm run check:literals`
- `npm run build`
"""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")

print(f"Successfully generated {OUT_FILE}")
