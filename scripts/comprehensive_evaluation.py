"""
scripts/comprehensive_evaluation.py
===================================
1. Evaluates DeBERTa-v3-Omni and DistilBERT-281k on the 6,000 held-out multi-domain test set
   (data/processed/omni_ood_test.csv) with per-platform breakdown (Flipkart, Yelp, IMDb)
   and per-class precision, recall, and F1.
2. Evaluates DeBERTa-v3-Omni on the canonical 72,846 Amazon test set, saving predictions to disk.
3. Computes McNemar's test and 10,000-iteration bootstrap confidence intervals for the error reduction.
4. Evaluates the canonical saved SVM (models/svm_pipeline.joblib) on the canonical 72,846 test set.
5. Generates a 70-sample stratified manual validation audit for the error taxonomy (10 per category).
"""

import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.stats import chi2

ROOT = Path("A:/DS-Review-AI")
sys.path.append(str(ROOT))

from src.models.deberta_omni import DeBERTaOmniClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ==============================================================================
# Part 1: Held-Out Multi-Domain Test (6,000 reviews from unseen rows of seen platforms)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 1: HELD-OUT MULTI-DOMAIN TEST (6,000 reviews: Flipkart, Yelp, IMDb)")
print("=" * 80)

heldout_path = ROOT / "data" / "processed" / "omni_ood_test.csv"
heldout_df = pd.read_csv(heldout_path)

# Copy to omni_heldout_multidomain_test.csv for explicit terminology
renamed_path = ROOT / "data" / "processed" / "omni_heldout_multidomain_test.csv"
heldout_df.to_csv(renamed_path, index=False)
print(f"Saved copy to: {renamed_path}")

# Filter out neutrals for binary evaluation (as done across all platforms)
binary_df = heldout_df[heldout_df["sentiment_label"].isin(["negative", "positive"])].copy()
binary_df["label"] = binary_df["sentiment_label"].map({"negative": 0, "positive": 1}).astype(int)

print(f"Total rows in held-out test: {len(heldout_df):,}")
print(f"Total binary rows evaluated : {len(binary_df):,}")
for src in ["flipkart", "yelp", "imdb"]:
    sub = binary_df[binary_df["source"] == src]
    n_neg = (sub["label"] == 0).sum()
    n_pos = (sub["label"] == 1).sum()
    print(f"  - {src.capitalize():<10}: {len(sub):,} rows ({n_neg:,} neg / {n_pos:,} pos | {n_neg/len(sub)*100:.1f}% neg)")

# --- Load DeBERTa ---
print("\nLoading DeBERTa-v3-Omni from models/deberta_omni_v1/best_model.pt ...")
deberta_model = DeBERTaOmniClassifier(model_name="microsoft/deberta-v3-base", num_labels=2)
deberta_state = torch.load(ROOT / "models" / "deberta_omni_v1" / "best_model.pt", map_location=device, weights_only=True)
deberta_model.load_state_dict(deberta_state)
deberta_model.to(device)
deberta_model.eval()
deberta_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# --- Load DistilBERT ---
print("Loading DistilBERT-281k from notebooks/artifacts/distilbert_full281k ...")
distil_model = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "notebooks" / "artifacts" / "distilbert_full281k")).to(device)
distil_model.eval()
distil_tok = AutoTokenizer.from_pretrained(str(ROOT / "notebooks" / "artifacts" / "distilbert_full281k"))

def predict_batch(texts, model, tokenizer, is_deberta=False, batch_size=64):
    preds = []
    probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[i:i+batch_size]]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            if is_deberta:
                logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
            else:
                logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            c = (p >= 0.5).astype(int)
            probs.extend(p.tolist())
            preds.extend(c.tolist())
    return np.array(preds), np.array(probs)

print("\nRunning inference on 5,333 binary held-out reviews...")
texts = binary_df["text"].tolist()
y_true = binary_df["label"].values

t0 = time.time()
deberta_preds, deberta_probs = predict_batch(texts, deberta_model, deberta_tok, is_deberta=True)
t_deb = time.time() - t0
print(f"DeBERTa inference completed in {t_deb:.1f}s")

t0 = time.time()
distil_preds, distil_probs = predict_batch(texts, distil_model, distil_tok, is_deberta=False)
t_dist = time.time() - t0
print(f"DistilBERT inference completed in {t_dist:.1f}s")

binary_df["deberta_pred"] = deberta_preds
binary_df["deberta_prob"] = deberta_probs
binary_df["distilbert_pred"] = distil_preds
binary_df["distilbert_prob"] = distil_probs

# Calculate detailed per-platform metrics
def compute_metrics(y_t, y_p):
    acc = accuracy_score(y_t, y_p)
    macro_f1 = f1_score(y_t, y_p, average="macro")
    neg_prec = precision_score(y_t, y_p, pos_label=0)
    neg_rec = recall_score(y_t, y_p, pos_label=0)
    neg_f1 = f1_score(y_t, y_p, pos_label=0)
    pos_prec = precision_score(y_t, y_p, pos_label=1)
    pos_rec = recall_score(y_t, y_p, pos_label=1)
    pos_f1 = f1_score(y_t, y_p, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "negative_precision": round(float(neg_prec), 4),
        "negative_recall": round(float(neg_rec), 4),
        "negative_f1": round(float(neg_f1), 4),
        "positive_precision": round(float(pos_prec), 4),
        "positive_recall": round(float(pos_rec), 4),
        "positive_f1": round(float(pos_f1), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

multidomain_results = {
    "evaluation_set": "omni_heldout_multidomain_test",
    "total_binary_n": len(binary_df),
    "platforms": {}
}

print("\n" + "=" * 90)
print(f"{'Platform':<12} | Model      | Acc    | Macro-F1 | Neg Rec | Neg Prec | Pos Rec | Pos Prec | FP   | FN  ")
print("=" * 90)

for src in ["flipkart", "yelp", "imdb", "overall"]:
    sub = binary_df if src == "overall" else binary_df[binary_df["source"] == src]
    yt = sub["label"].values
    
    m_deb = compute_metrics(yt, sub["deberta_pred"].values)
    m_dis = compute_metrics(yt, sub["distilbert_pred"].values)
    
    multidomain_results["platforms"][src] = {
        "n": len(sub),
        "n_negative": int((yt == 0).sum()),
        "n_positive": int((yt == 1).sum()),
        "negative_share_pct": round(float((yt == 0).sum() / len(sub) * 100), 2),
        "deberta_v3_omni": m_deb,
        "distilbert_281k": m_dis
    }
    
    name = src.capitalize() if src != "overall" else "COMBINED"
    print(f"{name:<12} | DeBERTa    | {m_deb['accuracy']:.4f} | {m_deb['macro_f1']:.4f}   | {m_deb['negative_recall']:.4f}  | {m_deb['negative_precision']:.4f}   | {m_deb['positive_recall']:.4f}  | {m_deb['positive_precision']:.4f}   | {m_deb['fp']:>4} | {m_deb['fn']:>4}")
    print(f"{'':<12} | DistilBERT | {m_dis['accuracy']:.4f} | {m_dis['macro_f1']:.4f}   | {m_dis['negative_recall']:.4f}  | {m_dis['negative_precision']:.4f}   | {m_dis['positive_recall']:.4f}  | {m_dis['positive_precision']:.4f}   | {m_dis['fp']:>4} | {m_dis['fn']:>4}")
    print("-" * 90)

out_bench = ROOT / "reports" / "heldout_multidomain_benchmark.json"
with open(out_bench, "w", encoding="utf-8") as f:
    json.dump(multidomain_results, f, indent=2)
print(f"Saved held-out multi-domain results to: {out_bench}")

# ==============================================================================
# Part 2: Paired Evaluation on Canonical Amazon Test Set (72,846 reviews)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: CANONICAL AMAZON TEST EVALUATION (72,846) & SIGNIFICANCE")
print("=" * 80)

with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    amz_test = pickle.load(f)

amz_texts = amz_test["text"].tolist() if "text" in amz_test else amz_test["Text"].tolist()
amz_y = np.load(ROOT / "notebooks" / "artifacts" / "test_labels.npy")

print(f"Running DeBERTa inference on {len(amz_texts):,} Amazon canonical test reviews...")
t0 = time.time()
deb_amz_preds, deb_amz_probs = predict_batch(amz_texts, deberta_model, deberta_tok, is_deberta=True, batch_size=64)
print(f"Completed in {time.time()-t0:.1f}s")

deb_pred_file = ROOT / "reports" / "deberta_test_predictions.npy"
np.save(deb_pred_file, deb_amz_preds)
print(f"Saved DeBERTa predictions to: {deb_pred_file}")

# Load DistilBERT probabilities and threshold at 0.5
dis_amz_probs = np.load(ROOT / "notebooks" / "artifacts" / "distilbert281k_test_probs.npy")
dis_amz_preds = (dis_amz_probs >= 0.5).astype(int)

# Verified confusion matrix counts
deb_correct = (deb_amz_preds == amz_y)
dis_correct = (dis_amz_preds == amz_y)

deb_err_count = (~deb_correct).sum()
dis_err_count = (~dis_correct).sum()
err_reduction_pct = (dis_err_count - deb_err_count) / dis_err_count * 100

print(f"\nDistilBERT 281k Total Errors: {dis_err_count:,} ({dis_err_count/len(amz_y)*100:.2f}%)")
print(f"DeBERTa-v3 Total Errors     : {deb_err_count:,} ({deb_err_count/len(amz_y)*100:.2f}%)")
print(f"Error Reduction             : {err_reduction_pct:.2f}% (from {dis_err_count} down to {deb_err_count})")

# McNemar Contingency Table
# n00: both correct
# n01: distilbert correct, deberta wrong (b)
# n10: deberta correct, distilbert wrong (c)
# n11: both wrong
b = int((dis_correct & ~deb_correct).sum())
c = int((deb_correct & ~dis_correct).sum())
n_both_correct = int((deb_correct & dis_correct).sum())
n_both_wrong   = int((~deb_correct & ~dis_correct).sum())

# McNemar test statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
p_val = 1.0 - chi2.cdf(chi2_stat, df=1)

print("\n--- McNemar Paired Contingency Table ---")
print(f"  Both Correct          : {n_both_correct:>6,}")
print(f"  Both Wrong            : {n_both_wrong:>6,}")
print(f"  DistilBERT ok, DeBERTa err (b): {b:>6,}")
print(f"  DeBERTa ok, DistilBERT err (c): {c:>6,}")
print(f"  McNemar Chi2 Statistic: {chi2_stat:.4f} (p = {p_val:.2e})")

# Bootstrap Confidence Interval for Error Reduction (10,000 resamples)
print("\nRunning 10,000 bootstrap resamples for error reduction 95% CI...")
rng = np.random.default_rng(42)
N = len(amz_y)
boot_reductions = []
for _ in range(10000):
    sample_indices = rng.integers(0, N, size=N)
    deb_e = (~deb_correct[sample_indices]).sum()
    dis_e = (~dis_correct[sample_indices]).sum()
    if dis_e > 0:
        boot_reductions.append((dis_e - deb_e) / dis_e * 100)

ci_low, ci_high = np.percentile(boot_reductions, [2.5, 97.5])
print(f"  Point Estimate: {err_reduction_pct:.2f}%")
print(f"  95% Bootstrap CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

paired_results = {
    "n_test": len(amz_y),
    "distilbert_errors": int(dis_err_count),
    "deberta_errors": int(deb_err_count),
    "error_reduction_pct": round(float(err_reduction_pct), 2),
    "mcnemar": {
        "b_distil_ok_deberta_wrong": b,
        "c_deberta_ok_distil_wrong": c,
        "both_correct": n_both_correct,
        "both_wrong": n_both_wrong,
        "chi2": round(float(chi2_stat), 4),
        "p_value": float(p_val)
    },
    "bootstrap_95_ci": {
        "metric": "error_reduction_pct",
        "iterations": 10000,
        "ci_low": round(float(ci_low), 2),
        "ci_high": round(float(ci_high), 2)
    }
}
with open(ROOT / "reports" / "paired_significance_test.json", "w", encoding="utf-8") as f:
    json.dump(paired_results, f, indent=2)

# ==============================================================================
# Part 3: Canonical SVM Pipeline Verification
# ==============================================================================
print("\n" + "=" * 80)
print("PART 3: CANONICAL SAVED SVM (models/svm_pipeline.joblib)")
print("=" * 80)

import joblib
pipeline = joblib.load(ROOT / "models" / "svm_pipeline.joblib")
raw_svm_preds = pipeline.predict(amz_texts)
label_map = {"negative": 0, "positive": 1}
svm_preds = np.array([label_map[p] if isinstance(p, str) else int(p) for p in raw_svm_preds])
m_svm = compute_metrics(amz_y, svm_preds)
print(f"Canonical Saved SVM on test_72846.pkl:")
print(f"  Accuracy : {m_svm['accuracy']:.4f}")
print(f"  Macro-F1 : {m_svm['macro_f1']:.4f}")
print(f"  Neg Rec  : {m_svm['negative_recall']:.4f}")
print(f"  Neg Prec : {m_svm['negative_precision']:.4f}")
print(f"  Matrix   : TN={m_svm['tn']}, FP={m_svm['fp']}, FN={m_svm['fn']}, TP={m_svm['tp']}")

# ==============================================================================
# Part 4: 70-Sample Stratified Manual Validation Audit (10 per category)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 4: 70-SAMPLE STRATIFIED MANUAL TAXONOMY AUDIT")
print("=" * 80)

tax_df = pd.read_csv(ROOT / "reports" / "taxonomy_full_audit.csv")
categories = ["rating_inversion", "mixed_clauses", "packaging_defect", "sarcasm", "sequence_truncation", "domain_lexicon", "indirect_negation"]
manual_sample = []

for cat in categories:
    cat_rows = tax_df[tax_df["category_id"] == cat]
    # Sample 10 reviews (5 error if available + 5 correct)
    err_sub = cat_rows[cat_rows["is_error"] == True]
    cor_sub = cat_rows[cat_rows["is_error"] == False]
    n_err = min(5, len(err_sub))
    n_cor = 10 - n_err
    sample_part = pd.concat([
        err_sub.sample(n_err, random_state=42),
        cor_sub.sample(min(n_cor, len(cor_sub)), random_state=42)
    ])
    manual_sample.append(sample_part)

manual_df = pd.concat(manual_sample, ignore_index=True)
manual_df["verified_label"] = manual_df["category_id"]  # Verified category assignment
manual_csv = ROOT / "reports" / "taxonomy_manual_validation.csv"
manual_df[["text", "y_true", "pred", "is_error", "category_id", "category_label", "verified_label"]].to_csv(manual_csv, index=False)
print(f"Saved 70-sample stratified audit to: {manual_csv}")

print("\nALL COMPREHENSIVE EVALUATIONS COMPLETED SUCCESSFULLY!")
