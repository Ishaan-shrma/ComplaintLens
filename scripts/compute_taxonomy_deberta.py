"""
scripts/compute_taxonomy_deberta.py
===================================
Executes Step 1 of the Error Taxonomy update:
- Runs the deterministic rule-based linguistic tagger across all 72,846 canonical test reviews.
- Uses the identical tagger rules and priority order from scripts/audit_taxonomy_baserate.py.
- Validates DistilBERT errors (1,901 total, 1367/340/101/36/25/20/12 breakdown).
- Validates DeBERTa raw errors (1,382 total).
- Verifies McNemar shared (968), fixed (933), and new (414) errors.
- Exports reports/taxonomy_deberta.csv, reports/taxonomy_distilbert.csv,
  and reports/taxonomy_full_audit_deberta.csv.
- Extracts verified example review IDs from DeBERTa errors for public/data/error_taxonomy.json.
"""

import sys
import pickle
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("A:/DS-Review-AI")
CSV_PATH = ROOT / "reports" / "test_predictions_per_row.csv"
PKL_PATH = ROOT / "notebooks" / "artifacts" / "test_72846.pkl"

print("=" * 80)
print("  STEP 1: RUN TAXONOMY TAGGER FOR DEBERTA-V3-OMNI AND DISTILBERT")
print("=" * 80)

# 1. Load data
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing {CSV_PATH}")
if not PKL_PATH.exists():
    raise FileNotFoundError(f"Missing {PKL_PATH}")

df_csv = pd.read_csv(CSV_PATH)
with open(PKL_PATH, "rb") as f:
    df_pkl = pickle.load(f)

# Ensure row alignment
assert len(df_csv) == len(df_pkl) == 72846, f"Length mismatch: {len(df_csv)} vs {len(df_pkl)}"
df_csv["text"] = df_pkl["text"].values
if "Id" in df_pkl.columns:
    df_csv["original_review_id"] = df_pkl["Id"].values
else:
    df_csv["original_review_id"] = df_csv["row_id"].astype(str)

# 2. Rule regex definitions (IDENTICAL to scripts/audit_taxonomy_baserate.py)
CONTRAST_RX = re.compile(r"\b(?:but|however|although|though|yet|nevertheless|despite|except)\b", re.I)
PACKAGING_RX = re.compile(r"\b(?:package|packaging|box|broken|crush|crushed|tear|torn|burst|leak|leaked|seal|sealed|shipping|shipment|transit|deliver|delivery)\b", re.I)
SARCASM_RX = re.compile(r"\b(?:yeah right|thanks for nothing|oh great|sure glad|waste of money|joke|laughable|unimpressed|what a shame)\b|!\?|\?\!|\.{3,}\s*(?:bad|terrible|awful|horrible)", re.I)
INVERSION_POS_RX = re.compile(r"\b(?:delicious|love it|great|best ever|amazing|wonderful|excellent|favorite)\b", re.I)
INVERSION_NEG_RX = re.compile(r"\b(?:horrible|terrible|awful|inedible|disgusting|vomit|nasty|waste of money|hate it)\b", re.I)
FOOD_IDIOM_RX = re.compile(r"\b(?:stale|rancid|chemical|cardboard|chalky|bland|rubbery|bitter|aftertaste|sour|spoil|spoiled|expired|mold|moldy)\b", re.I)

def assign_category(row):
    text = str(row["text"])
    words = text.split()
    y = row["y_true"]
    
    # Priority 1: Rating-Proxy Inversion
    if y == 0 and len(INVERSION_POS_RX.findall(text)) >= 2 and not CONTRAST_RX.search(text):
        return "rating_inversion", "Rating-Text Inversion (Label Noise)"
    if y == 1 and len(INVERSION_NEG_RX.findall(text)) >= 2 and not CONTRAST_RX.search(text):
        return "rating_inversion", "Rating-Text Inversion (Label Noise)"
        
    # Priority 2: Mixed / Contrastive Clauses
    if CONTRAST_RX.search(text):
        return "mixed_clauses", "Mixed / Contrastive Clauses ('good, but...')"
        
    # Priority 3: Packaging & Transit Defects
    if PACKAGING_RX.search(text):
        return "packaging_defect", "Subtle Packaging/Transit Complaints"
        
    # Priority 4: Sarcasm & Ironic Polarity
    if SARCASM_RX.search(text):
        return "sarcasm", "Sarcasm & Ironic Polarity"
        
    # Priority 5: Sequence Truncation (>= 128 words)
    if len(words) >= 128:
        return "sequence_truncation", "Sequence Truncation Beyond 128 Tokens"
        
    # Priority 6: Food & Domain Idioms
    if FOOD_IDIOM_RX.search(text):
        return "domain_lexicon", "Domain-Specific Food Idioms & Jargon"
        
    # Priority 7: Indirect / Nuanced Grievance (residual)
    return "indirect_negation", "Indirect & Nuanced Sentiment (Residual)"

print("\n[1/4] Tagging all 72,846 canonical test reviews...")
cats = []
labels = []
for idx, row in df_csv.iterrows():
    c_id, c_lbl = assign_category(row)
    cats.append(c_id)
    labels.append(c_lbl)

df_csv["category_id"] = cats
df_csv["category_label"] = labels

# Error flags
df_csv["distilbert_error"] = (df_csv["distilbert_pred"] != df_csv["y_true"])
df_csv["deberta_raw_error"] = (df_csv["deberta_raw_pred"] != df_csv["y_true"])

# Sanity checks
n_distil_err = int(df_csv["distilbert_error"].sum())
n_deberta_raw_err = int(df_csv["deberta_raw_error"].sum())

print(f"\nTotal test reviews:            {len(df_csv):,}")
print(f"DistilBERT total errors:       {n_distil_err}")
print(f"DeBERTa raw total errors:      {n_deberta_raw_err}")

assert n_distil_err == 1901, f"FAIL: Expected DistilBERT errors == 1901, got {n_distil_err}"
assert n_deberta_raw_err == 1382, f"FAIL: Expected DeBERTa raw errors == 1382, got {n_deberta_raw_err}"

cat_order = [
    "mixed_clauses",
    "indirect_negation",
    "packaging_defect",
    "sequence_truncation",
    "rating_inversion",
    "domain_lexicon",
    "sarcasm"
]

expected_distil_counts = {
    "mixed_clauses": 1367,
    "indirect_negation": 340,
    "packaging_defect": 101,
    "sequence_truncation": 36,
    "rating_inversion": 25,
    "domain_lexicon": 20,
    "sarcasm": 12
}

print("\n[2/4] Verifying Category Counts:")
print("-" * 80)
print(f"{'Category ID':<22} | {'Total Reviews':<13} | {'DistilBERT Err':<14} | {'DeBERTa Err':<12} | {'DeBERTa Rate'}")
print("-" * 80)

deberta_counts = {}
distil_counts = {}
total_reviews_by_cat = {}

for cid in cat_order:
    sub = df_csv[df_csv["category_id"] == cid]
    tot = len(sub)
    d_cnt = int(sub["distilbert_error"].sum())
    deb_cnt = int(sub["deberta_raw_error"].sum())
    deb_rate = (deb_cnt / tot) * 100.0

    total_reviews_by_cat[cid] = tot
    distil_counts[cid] = d_cnt
    deberta_counts[cid] = deb_cnt

    exp_d = expected_distil_counts[cid]
    assert d_cnt == exp_d, f"FAIL: DistilBERT count for {cid} was {d_cnt}, expected {exp_d}"
    print(f"{cid:<22} | {tot:>13,d} | {d_cnt:>14,d} | {deb_cnt:>12,d} | {deb_rate:>10.2f}%")

print("-" * 80)
print(f"{'TOTAL':<22} | {len(df_csv):>13,d} | {sum(distil_counts.values()):>14,d} | {sum(deberta_counts.values()):>12,d}")
print("-" * 80)

assert sum(distil_counts.values()) == 1901, "DistilBERT categories do not sum to 1901"
assert sum(deberta_counts.values()) == 1382, "DeBERTa categories do not sum to 1382"
print("[SANITY CHECK PASSED] All category counts exactly match expected discrete numbers.")

# 3. Overlap / McNemar analysis
both_wrong = int((df_csv["distilbert_error"] & df_csv["deberta_raw_error"]).sum())
distil_only = int((df_csv["distilbert_error"] & ~df_csv["deberta_raw_error"]).sum())
deberta_only = int((~df_csv["distilbert_error"] & df_csv["deberta_raw_error"]).sum())
both_correct = int((~df_csv["distilbert_error"] & ~df_csv["deberta_raw_error"]).sum())

print("\n[3/4] McNemar Overlap Decomposition:")
print(f"  Shared errors (both models wrong):      {both_wrong}")
print(f"  Fixed errors (DistilBERT wrong, DeBERTa correct): {distil_only} (c)")
print(f"  New errors (DistilBERT correct, DeBERTa wrong):   {deberta_only} (b)")
print(f"  Both correct:                           {both_correct}")
assert both_wrong == 968, f"Expected shared errors == 968, got {both_wrong}"
assert distil_only == 933, f"Expected fixed errors == 933, got {distil_only}"
assert deberta_only == 414, f"Expected new errors == 414, got {deberta_only}"
print("  -> Verified McNemar counts: b = 414, c = 933, shared = 968. Net error reduction = 519 (1,901 - 1,382).")

# 4. Save CSV Reports
reports_dir = ROOT / "reports"
deb_err_df = df_csv[df_csv["deberta_raw_error"]].copy()
dis_err_df = df_csv[df_csv["distilbert_error"]].copy()

deb_err_df.to_csv(reports_dir / "taxonomy_deberta.csv", index=False)
dis_err_df.to_csv(reports_dir / "taxonomy_distilbert.csv", index=False)
df_csv.to_csv(reports_dir / "taxonomy_full_audit_deberta.csv", index=False)

print("\n[4/4] Saved CSV Reports:")
print(f"  - {reports_dir / 'taxonomy_deberta.csv'} ({len(deb_err_df)} rows)")
print(f"  - {reports_dir / 'taxonomy_distilbert.csv'} ({len(dis_err_df)} rows)")
print(f"  - {reports_dir / 'taxonomy_full_audit_deberta.csv'} ({len(df_csv)} rows)")

# 5. Extract and verify example review IDs from DeBERTa error list
print("\n[5/5] Extracting Verified DeBERTa Error Example IDs per category:")
example_ids_by_cat = {}
for cid in cat_order:
    sub_err = deb_err_df[deb_err_df["category_id"] == cid]
    # Pick first 3 review IDs
    ex_ids = [str(r) for r in sub_err["row_id"].iloc[:3].tolist()]
    example_ids_by_cat[cid] = ex_ids
    # Verify each ID is indeed in that category and is a DeBERTa error
    for ex_id in ex_ids:
        row_check = df_csv[df_csv["row_id"] == int(ex_id)].iloc[0]
        assert row_check["category_id"] == cid, f"Example ID {ex_id} category mismatch: {row_check['category_id']} != {cid}"
        assert row_check["deberta_raw_error"] == True, f"Example ID {ex_id} is not a DeBERTa error"
    print(f"  {cid:<22}: {ex_ids} (All 3 verified)")

print("\nSTEP 1 COMPLETE WITH ZERO ERRORS.")
