"""
Recomputes McNemar, Paired Row-Wise Bootstrap (10k iterations), and Token-Length Slices
directly from reports/test_predictions_per_row.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm

ROOT = Path("A:/DS-Review-AI")
csv_path = ROOT / "reports" / "test_predictions_per_row.csv"
df = pd.read_csv(csv_path)

N = len(df)
y_true = df["y_true"].to_numpy()
dis_pred = df["distilbert_pred"].to_numpy()
raw_pred = df["deberta_raw_pred"].to_numpy()
norm_pred = df["deberta_norm_pred"].to_numpy()
tok_len = df["token_len_distilbert"].to_numpy()

dis_err = (dis_pred != y_true)
raw_err = (raw_pred != y_true)
norm_err = (norm_pred != y_true)

def analyze_pair(name: str, pred_err: np.ndarray, base_err: np.ndarray):
    # Base = DistilBERT, Model = DeBERTa variant
    # b: Base correct, Model wrong
    # c: Model correct, Base wrong
    b = int(((~base_err) & pred_err).sum())
    c = int((base_err & (~pred_err)).sum())
    both_correct = int(((~base_err) & (~pred_err)).sum())
    both_wrong = int((base_err & pred_err).sum())
    
    n_base_err = int(base_err.sum())
    n_model_err = int(pred_err.sum())
    point_reduction = (n_base_err - n_model_err) / n_base_err * 100
    
    # McNemar with continuity correction
    mcnemar_chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = float(1.0 - chi2.cdf(mcnemar_chi2, df=1))
    
    # Asymptotic Delta Method for Relative Error Reduction: R = (c - b) / D
    # where D = b_wrong + c = base_err.sum()
    # Var(c - b) = b + c
    # Var(D) = D * (1 - D/N)
    # Cov(c - b, D) = Var(c) = c * (1 - c/N)
    # Using delta method: Var(R) = (1/D)^2 * Var(c - b) + ((c - b)/D^2)^2 * Var(D) - 2 * ((c - b)/D^3) * Cov(c - b, D)
    D = float(n_base_err)
    diff = float(c - b)
    var_diff = float(b + c)
    var_D = D * (1.0 - D / N)
    cov_diff_D = float(c) * (1.0 - float(c) / N)
    
    var_R = (1.0 / (D ** 2)) * var_diff + ((diff ** 2) / (D ** 4)) * var_D - 2.0 * (diff / (D ** 3)) * cov_diff_D
    se_R = np.sqrt(max(0.0, var_R)) * 100
    asymp_ci_low = point_reduction - 1.96 * se_R
    asymp_ci_high = point_reduction + 1.96 * se_R
    
    # Paired row-wise bootstrap (10,000 resamples)
    rng = np.random.default_rng(42)
    boot_reductions = []
    for _ in range(10000):
        idx = rng.integers(0, N, size=N)
        b_err = base_err[idx].sum()
        m_err = pred_err[idx].sum()
        if b_err > 0:
            boot_reductions.append((b_err - m_err) / b_err * 100)
            
    pct_ci_low, pct_ci_high = np.percentile(boot_reductions, [2.5, 97.5])
    boot_se = np.std(boot_reductions)
    
    return {
        "name": name,
        "n_base_err": n_base_err,
        "n_model_err": n_model_err,
        "point_reduction_pct": round(point_reduction, 2),
        "mcnemar": {
            "b_distil_ok_model_wrong": b,
            "c_model_ok_distil_wrong": c,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "chi2": round(float(mcnemar_chi2), 4),
            "p_val": p_val
        },
        "bootstrap_10k": {
            "ci_low": round(float(pct_ci_low), 2),
            "ci_high": round(float(pct_ci_high), 2),
            "boot_se": round(float(boot_se), 3),
            "margin_pct": round(float(1.96 * boot_se), 2)
        },
        "asymptotic_delta_method": {
            "se": round(float(se_R), 3),
            "ci_low": round(float(asymp_ci_low), 2),
            "ci_high": round(float(asymp_ci_high), 2),
            "margin_pct": round(float(1.96 * se_R), 2)
        }
    }

raw_res = analyze_pair("DistilBERT vs DeBERTa Raw (1,382 errors)", raw_err, dis_err)
norm_res = analyze_pair("DistilBERT vs DeBERTa Normalized (1,361 errors)", norm_err, dis_err)

print("=== BOOTSTRAP AND SIGNIFICANCE RESULTS ===")
print(json.dumps(raw_res, indent=2))
print(json.dumps(norm_res, indent=2))

# Length slice analysis
long_mask = (tok_len > 128)
short_mask = ~long_mask

n_long = int(long_mask.sum())
n_short = int(short_mask.sum())

print("\n=== TOKEN LENGTH SLICE ANALYSIS (DistilBERT Tokenizer, cut=128) ===")
print(f"Total reviews >128 tokens : {n_long:,} ({n_long/N*100:.2f}%)")
print(f"Total reviews <=128 tokens: {n_short:,} ({n_short/N*100:.2f}%)")

def slice_stats(mask, mask_name):
    n_sub = int(mask.sum())
    dis_e = int(dis_err[mask].sum())
    raw_e = int(raw_err[mask].sum())
    norm_e = int(norm_err[mask].sum())
    return {
        "slice": mask_name,
        "n": n_sub,
        "distilbert": { "errors": dis_e, "rate_pct": round(dis_e / n_sub * 100, 2) },
        "deberta_norm": { "errors": norm_e, "rate_pct": round(norm_e / n_sub * 100, 2), "reduction_pct": round((dis_e - norm_e)/dis_e * 100, 2) },
        "deberta_raw": { "errors": raw_e, "rate_pct": round(raw_e / n_sub * 100, 2), "reduction_pct": round((dis_e - raw_e)/dis_e * 100, 2) },
    }

long_slice = slice_stats(long_mask, ">128 tokens")
short_slice = slice_stats(short_mask, "<=128 tokens")

print(json.dumps(long_slice, indent=2))
print(json.dumps(short_slice, indent=2))

# Save analysis to json
with open(ROOT / "reports" / "recomputed_bootstrap_and_length_slices.json", "w", encoding="utf-8") as f:
    json.dump({
        "raw_comparison": raw_res,
        "normalized_comparison": norm_res,
        "length_slices": {
            "long_reviews_over_128": long_slice,
            "short_reviews_under_eq_128": short_slice
        }
    }, f, indent=2)
print("\nSaved full recomputation report to reports/recomputed_bootstrap_and_length_slices.json")
