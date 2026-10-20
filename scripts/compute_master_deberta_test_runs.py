"""
Computes exact DeBERTa probabilities and predictions on the 72,846 canonical test set:
1. Raw strings (no normalization)
2. Normalized with static padding (max_length=128, padding='max_length')
3. Normalized with dynamic padding (max_length=128, padding=True)

Computes exact ROC-AUC, confusion matrices, McNemar chi2, and saves master per-row CSV.
"""
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import chi2

ROOT = Path("A:/DS-Review-AI")
import sys
sys.path.append(str(ROOT))

from src.models.deberta_omni import DeBERTaOmniClassifier
from src.data.normalizer import batch_normalize
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# 1. Load test data
with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    test_df = pickle.load(f)

raw_texts = test_df["text"].tolist() if "text" in test_df else test_df["Text"].tolist()
y_true = np.load(ROOT / "notebooks" / "artifacts" / "test_labels.npy")
N = len(y_true)

# 2. Load model & tokenizer
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DeBERTaOmniClassifier("microsoft/deberta-v3-base", 2)
model.load_state_dict(torch.load(ROOT / "models" / "deberta_omni_v1" / "best_model.pt", map_location=device, weights_only=True))
model.to(device).eval()

def run_inference(texts, padding_mode, batch_size=128):
    probs_list = []
    preds_list = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=padding_mode, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred = (p >= 0.5).astype(int)
            probs_list.extend(p.tolist())
            preds_list.extend(pred.tolist())
    print(f"Inference ({padding_mode}) completed in {time.time()-t0:.1f}s")
    return np.array(probs_list), np.array(preds_list)

# Run 1: Raw strings
print("\n--- Run 1: Raw Text Strings ---")
raw_probs, raw_preds = run_inference(raw_texts, padding_mode=True, batch_size=128)
tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_true, raw_preds).ravel()
auc_r = roc_auc_score(y_true, raw_probs)
print(f"Raw: TN={tn_r}, FP={fp_r}, FN={fn_r}, TP={tp_r}, Total Errors={fp_r+fn_r}, ROC-AUC={auc_r:.4f}")

# Run 2: Normalized texts
print("\nNormalizing texts via batch_normalize()...")
norm_texts = batch_normalize(raw_texts)

# Run 2a: Normalized Dynamic Padding
print("\n--- Run 2a: Normalized Dynamic Padding ---")
norm_dyn_probs, norm_dyn_preds = run_inference(norm_texts, padding_mode=True, batch_size=128)
tn_nd, fp_nd, fn_nd, tp_nd = confusion_matrix(y_true, norm_dyn_preds).ravel()
auc_nd = roc_auc_score(y_true, norm_dyn_probs)
print(f"Norm Dynamic: TN={tn_nd}, FP={fp_nd}, FN={fn_nd}, TP={tp_nd}, Total Errors={fp_nd+fn_nd}, ROC-AUC={auc_nd:.4f}")

# Run 2b: Normalized Static Padding
print("\n--- Run 2b: Normalized Static Padding (max_length=128) ---")
norm_stat_probs, norm_stat_preds = run_inference(norm_texts, padding_mode="max_length", batch_size=128)
tn_ns, fp_ns, fn_ns, tp_ns = confusion_matrix(y_true, norm_stat_preds).ravel()
auc_ns = roc_auc_score(y_true, norm_stat_probs)
print(f"Norm Static: TN={tn_ns}, FP={fp_ns}, FN={fn_ns}, TP={tp_ns}, Total Errors={fp_ns+fn_ns}, ROC-AUC={auc_ns:.4f}")

# Load DistilBERT
dis_probs = np.load(ROOT / "notebooks" / "artifacts" / "distilbert281k_test_probs.npy")
dis_preds = (dis_probs >= 0.5).astype(int)
auc_dis = roc_auc_score(y_true, dis_probs)
tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, dis_preds).ravel()
print(f"\nDistilBERT: TN={tn_d}, FP={fp_d}, FN={fn_d}, TP={tp_d}, Total Errors={fp_d+fn_d}, ROC-AUC={auc_dis:.4f}")

# McNemar recomputations vs DistilBERT
def compute_mcnemar(m_preds, m_name):
    dis_err = (dis_preds != y_true)
    m_err = (m_preds != y_true)
    b = int(((~dis_err) & m_err).sum())
    c = int((dis_err & (~m_err)).sum())
    both_correct = int(((~dis_err) & (~m_err)).sum())
    both_wrong = int((dis_err & m_err).sum())
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = 1.0 - chi2.cdf(chi2_stat, df=1)
    red = (dis_err.sum() - m_err.sum()) / dis_err.sum() * 100
    print(f"McNemar {m_name}: b={b}, c={c}, both_correct={both_correct}, both_wrong={both_wrong}, chi2={chi2_stat:.4f}, p={p_val:.2e}, red={red:.2f}%")
    return b, c, both_correct, both_wrong, chi2_stat, p_val, red

b_r, c_r, bc_r, bw_r, chi2_r, p_r, red_r = compute_mcnemar(raw_preds, "Raw")
b_nd, c_nd, bc_nd, bw_nd, chi2_nd, p_nd, red_nd = compute_mcnemar(norm_dyn_preds, "Norm Dynamic")
b_ns, c_ns, bc_ns, bw_ns, chi2_ns, p_ns, red_ns = compute_mcnemar(norm_stat_preds, "Norm Static")

# Save numpy arrays
np.save(ROOT / "reports" / "deberta_test_probs_raw.npy", raw_probs)
np.save(ROOT / "reports" / "deberta_test_probs_norm_dynamic.npy", norm_dyn_probs)
np.save(ROOT / "reports" / "deberta_test_probs_norm_static.npy", norm_stat_probs)
np.save(ROOT / "reports" / "deberta_test_predictions_norm_static.npy", norm_stat_preds)

# DistilBERT token lengths
tok_dis = AutoTokenizer.from_pretrained(ROOT / "notebooks" / "artifacts" / "distilbert_full281k")
tok_lens_dis = [len(tok_dis.encode(t, truncation=False)) for t in raw_texts]
word_lens = [len(t.split()) for t in raw_texts]

# Save comprehensive per-row CSV
out_df = pd.DataFrame({
    "row_id": np.arange(N),
    "y_true": y_true,
    "distilbert_p_pos": np.round(dis_probs, 5),
    "distilbert_pred": dis_preds,
    "deberta_raw_p_pos": np.round(raw_probs, 5),
    "deberta_raw_pred": raw_preds,
    "deberta_norm_dyn_p_pos": np.round(norm_dyn_probs, 5),
    "deberta_norm_dyn_pred": norm_dyn_preds,
    "deberta_norm_stat_p_pos": np.round(norm_stat_probs, 5),
    "deberta_norm_stat_pred": norm_stat_preds,
    "token_len_distilbert": tok_lens_dis,
    "word_len": word_lens,
    "differs_raw_vs_norm_dyn": (raw_preds != norm_dyn_preds).astype(int),
    "differs_raw_vs_norm_stat": (raw_preds != norm_stat_preds).astype(int),
})

csv_file = ROOT / "reports" / "test_predictions_per_row.csv"
out_df.to_csv(csv_file, index=False)
print(f"\nSaved master per-row file to {csv_file} ({len(out_df):,} rows)")
