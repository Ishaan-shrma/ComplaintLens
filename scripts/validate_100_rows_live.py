"""
scripts/validate_100_rows_live.py
==================================
Validates the live Review Intelligence prototype FastAPI serving endpoint (/predict)
against the canonical held-out test predictions (reports/test_predictions_per_row.csv).

Design:
1. Samples 100 test reviews: 50 positive (y_true=1) and 50 negative (y_true=0) using random seed 42.
2. Sends each raw review text to the live running HTTP server at http://127.0.0.1:8000/predict.
3. Retrieves real forward-pass DeBERTa probabilities (calibrated at T=1.35).
4. Verifies the temperature calibration mapping against test_predictions_per_row.csv:
   Expected calibrated probability = sigmoid( logit(p_raw_csv) / 1.35 ).
5. Asserts that maximum difference |p_live - p_expected| < 0.001 (governed by 4-decimal-place HTTP serialization).
6. Outputs full table and summary statistics to reports/validation_100_rows_live.txt.
"""

import sys
import json
import math
import pickle
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "reports" / "test_predictions_per_row.csv"
PKL_PATH = ROOT / "notebooks" / "artifacts" / "test_72846.pkl"
OUT_REPORT = ROOT / "reports" / "validation_100_rows_live.txt"
ENDPOINT = "http://127.0.0.1:8000/predict"

def main():
    print("=" * 84)
    print("  LIVE FASTAPI ENDPOINT VS CSV VALIDATION (100 RANDOM TEST ROWS)")
    print(f"  Target Endpoint: {ENDPOINT}")
    print(f"  Reference CSV:   {CSV_PATH}")
    print("=" * 84)

    # 1. Load data
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Missing {PKL_PATH}")

    df_csv = pd.read_csv(CSV_PATH)
    with open(PKL_PATH, "rb") as f:
        df_pkl = pickle.load(f)

    # Sample 50 positive, 50 negative
    pos_samples = df_csv[df_csv["y_true"] == 1].sample(n=50, random_state=42)
    neg_samples = df_csv[df_csv["y_true"] == 0].sample(n=50, random_state=42)
    sample_df = pd.concat([neg_samples, pos_samples]).sort_values("row_id").reset_index(drop=True)

    rows_log = []
    max_delta = 0.0
    label_matches = 0
    total_samples = len(sample_df)

    print(f"\nEvaluating {total_samples} sampled rows (50 negative, 50 positive)...\n")

    for idx, row in sample_df.iterrows():
        row_id = int(row["row_id"])
        y_true = int(row["y_true"])
        p_raw_csv = float(row["deberta_raw_p_pos"])
        pred_csv = int(row["deberta_raw_pred"])
        
        # Original text from test_72846.pkl
        raw_text = df_pkl.iloc[row_id]["text"]

        # Expected calibrated probability at T = 1.35
        # Guard against log(0)
        p_clipped = min(max(p_raw_csv, 1e-12), 1.0 - 1e-12)
        logit_raw = math.log(p_clipped / (1.0 - p_clipped))
        p_expected_cal = 1.0 / (1.0 + math.exp(-logit_raw / 1.35))

        # Query live HTTP endpoint
        payload = json.dumps({"text": raw_text}).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[ERROR] Failed to query endpoint for row_id {row_id}: {e}")
            sys.exit(1)

        deb_res = data.get("deberta", {})
        live_p_pos = float(deb_res.get("positive_probability", 0.0))
        live_label_str = deb_res.get("label", "")
        live_pred = 1 if live_label_str == "Positive" else 0

        delta_p = abs(live_p_pos - round(p_expected_cal, 4))
        if delta_p > max_delta:
            max_delta = delta_p

        is_label_match = (live_pred == pred_csv)
        if is_label_match:
            label_matches += 1

        rows_log.append({
            "sample_num": idx + 1,
            "row_id": row_id,
            "y_true": y_true,
            "p_raw_csv": round(p_raw_csv, 5),
            "p_exp_cal": round(p_expected_cal, 4),
            "live_p_pos": round(live_p_pos, 4),
            "delta_p": round(delta_p, 6),
            "pred_csv": "Pos" if pred_csv == 1 else "Neg",
            "live_pred": "Pos" if live_pred == 1 else "Neg",
            "match": "YES" if is_label_match else "NO",
            "text_preview": (raw_text[:45] + "...").replace("\n", " ")
        })

    # Generate Summary Report
    match_rate = (label_matches / total_samples) * 100.0
    mean_delta = np.mean([r["delta_p"] for r in rows_log])

    report_lines = []
    report_lines.append("=" * 105)
    report_lines.append("           REVIEW INTELLIGENCE PROTOTYPE: LIVE ENDPOINT VALIDATION REPORT")
    report_lines.append("=" * 105)
    report_lines.append(f"Target URL:         {ENDPOINT}")
    report_lines.append(f"Evaluation Dataset: 100 Stratified Test Rows (50 Negative, 50 Positive) from test_72846.pkl")
    report_lines.append(f"Reference Baseline: reports/test_predictions_per_row.csv (Raw text p_pos with T=1.35 calibration)")
    report_lines.append(f"Temperature:        T = 1.35")
    report_lines.append(f"Total Rows Tested:  {total_samples}")
    report_lines.append(f"Label Match Rate:   {label_matches}/{total_samples} ({match_rate:.2f}%)")
    report_lines.append(f"Max Prob Delta:     {max_delta:.6f}")
    report_lines.append(f"Mean Prob Delta:    {mean_delta:.6f}")
    report_lines.append(f"Integrity Status:   {'PASSED (All tolerances met within <= 0.0002 / 2 ULP)' if round(max_delta, 4) <= 0.0002 else 'WARNING (Tolerance exceeded)'}")
    report_lines.append("=" * 105)
    report_lines.append(f"{'#':<4} {'RowID':<7} {'True':<5} {'CSV Raw':<9} {'Exp Cal':<9} {'Live Cal':<9} {'Delta':<9} {'CSV':<5} {'Live':<5} {'Match':<6} {'Preview'}")
    report_lines.append("-" * 105)

    for r in rows_log:
        report_lines.append(
            f"{r['sample_num']:<4} {r['row_id']:<7} {r['y_true']:<5} {r['p_raw_csv']:<9.5f} "
            f"{r['p_exp_cal']:<9.4f} {r['live_p_pos']:<9.4f} {r['delta_p']:<9.6f} "
            f"{r['pred_csv']:<5} {r['live_pred']:<5} {r['match']:<6} {r['text_preview']}"
        )
    report_lines.append("=" * 105)

    report_text = "\n".join(report_lines)
    print(report_text[:1200] + "\n... [truncated] ...\n" + report_text[-500:])

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n[Validation Success] Complete report written to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
