"""
Generates per-row prediction file for canonical test set (N=72,846).
Includes true label, probabilities, and binary predictions for DistilBERT and DeBERTa variants,
along with token lengths and divergence indicators.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path("A:/DS-Review-AI")

print("[1/5] Loading test dataset and true labels...")
with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    test_df = pickle.load(f)

texts = test_df["text"].tolist() if "text" in test_df else test_df["Text"].tolist()
y_true = np.load(ROOT / "notebooks" / "artifacts" / "test_labels.npy")

print("[2/5] Loading model predictions...")
dis_probs = np.load(ROOT / "notebooks" / "artifacts" / "distilbert281k_test_probs.npy")
dis_preds = (dis_probs >= 0.5).astype(int)

deb_raw_preds = np.load(ROOT / "reports" / "deberta_test_predictions.npy")
deb_norm_preds = np.load(ROOT / "reports" / "deberta_test_predictions_normalized.npy")

print("[3/5] Computing token lengths via DistilBERT WordPiece tokenizer...")
tok_dis = AutoTokenizer.from_pretrained(ROOT / "notebooks" / "artifacts" / "distilbert_full281k")
# Compute token lengths
tok_lens_dis = [len(tok_dis.encode(t, truncation=False)) for t in texts]
word_lens = [len(t.split()) for t in texts]

print("[4/5] Constructing DataFrame...")
out_df = pd.DataFrame({
    "row_id": np.arange(len(y_true)),
    "y_true": y_true,
    "distilbert_p_pos": np.round(dis_probs, 5),
    "distilbert_pred": dis_preds,
    "deberta_raw_pred": deb_raw_preds,
    "deberta_norm_pred": deb_norm_preds,
    "token_len_distilbert": tok_lens_dis,
    "word_len": word_lens,
    "pred_differs_raw_vs_norm": (deb_raw_preds != deb_norm_preds).astype(int)
})

out_path_csv = ROOT / "reports" / "test_predictions_per_row.csv"
out_df.to_csv(out_path_csv, index=False)
print(f"[5/5] Saved per-row predictions to {out_path_csv} ({len(out_df):,} rows)")

# Summary stats
print("\n=== SUMMARY RECOMPUTATION ===")
print(f"DistilBERT errors: {(dis_preds != y_true).sum()} (FP: {((dis_preds == 1) & (y_true == 0)).sum()}, FN: {((dis_preds == 0) & (y_true == 1)).sum()})")
print(f"DeBERTa raw errors: {(deb_raw_preds != y_true).sum()} (FP: {((deb_raw_preds == 1) & (y_true == 0)).sum()}, FN: {((deb_raw_preds == 0) & (y_true == 1)).sum()})")
print(f"DeBERTa norm errors: {(deb_norm_preds != y_true).sum()} (FP: {((deb_norm_preds == 1) & (y_true == 0)).sum()}, FN: {((deb_norm_preds == 0) & (y_true == 1)).sum()})")
print(f"Rows where raw vs norm differ: {out_df['pred_differs_raw_vs_norm'].sum()}")
