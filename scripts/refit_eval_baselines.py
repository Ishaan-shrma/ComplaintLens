import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

ROOT = Path("A:/DS-Review-AI")

print("[1/4] Loading canonical training set (281,271) and test set (72,846)...")
with open(ROOT / "notebooks" / "artifacts" / "full_train_281271.pkl", "rb") as f:
    train_df = pickle.load(f)
with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    test_df = pickle.load(f)

X_train = train_df["text"].tolist() if "text" in train_df else train_df["Text"].tolist()
y_train = train_df["label"].astype(int).values

X_test = test_df["text"].tolist() if "text" in test_df else test_df["Text"].tolist()
y_test = test_df["label"].astype(int).values

print(f"Train: {len(X_train):,} reviews | Neg: {(y_train == 0).sum():,}, Pos: {(y_train == 1).sum():,}")
print(f"Test : {len(X_test):,} reviews  | Neg: {(y_test == 0).sum():,}, Pos: {(y_test == 1).sum():,}")

print("\n[2/4] Fitting tuned TF-IDF Vectorizer (ngram=(1,3), max_features=40,000, sublinear_tf=True)...")
t0 = time.time()
tfidf = TfidfVectorizer(max_features=40000, min_df=5, ngram_range=(1, 3), sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print(f"TF-IDF fitted in {time.time()-t0:.1f}s. Vocabulary size: {X_train_vec.shape[1]:,}")

models = {
    "Linear SVM (C=0.5, Tuned TF-IDF)": LinearSVC(C=0.5, max_iter=2000, random_state=42),
    "Logistic Regression (L2, C=1.0)": LogisticRegression(max_iter=1000, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    "Majority Dummy Classifier": DummyClassifier(strategy="most_frequent")
}

results = []
print("\n[3/4] Fitting and evaluating baselines on canonical test set...")
for name, clf in models.items():
    t_m = time.time()
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    fit_time = time.time() - t_m
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    neg_rec = recall_score(y_test, preds, pos_label=0)
    neg_prec = precision_score(y_test, preds, pos_label=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    print(f"  {name:<35} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | NegRec: {neg_rec:.4f} | NegPrec: {neg_prec:.4f} | FP: {fp:>5} | FN: {fn:>5} | Time: {fit_time:.1f}s")
    results.append({
        "model_id": name,
        "display_name": name,
        "train_n": len(X_train),
        "eval_set_id": "amazon_test",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "negative_precision": round(float(neg_prec), 4),
        "negative_recall": round(float(neg_rec), 4),
        "fp": int(fp),
        "fn": int(fn),
        "status": "evaluated once"
    })

# Add VADER on the exact same test set
print("\n[4/4] Evaluating VADER Lexicon on canonical test set...")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
vader_preds = []
for t in X_test:
    vader_preds.append(1 if analyzer.polarity_scores(t)["compound"] >= 0 else 0)
vader_preds = np.array(vader_preds)
acc_v = accuracy_score(y_test, vader_preds)
macro_f1_v = f1_score(y_test, vader_preds, average="macro")
neg_rec_v = recall_score(y_test, vader_preds, pos_label=0)
neg_prec_v = precision_score(y_test, vader_preds, pos_label=0)
tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_test, vader_preds).ravel()
print(f"  {'VADER Lexicon (Heuristic Rule)':<35} | Acc: {acc_v:.4f} | Macro-F1: {macro_f1_v:.4f} | NegRec: {neg_rec_v:.4f} | NegPrec: {neg_prec_v:.4f} | FP: {fp_v:>5} | FN: {fn_v:>5}")

# Save refit baseline results to reports/canonical_baseline_evaluations.json
out_file = ROOT / "reports" / "canonical_baseline_evaluations.json"
import json
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved canonical baseline evaluations to: {out_file}")
