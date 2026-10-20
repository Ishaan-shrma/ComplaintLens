import pickle
import numpy as np
import pandas as pd
import re
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")

# 1. Load test set and predictions
with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    df = pickle.load(f)

y_true = np.load(ROOT / "notebooks" / "artifacts" / "test_labels.npy")
probs = np.load(ROOT / "notebooks" / "artifacts" / "distilbert281k_test_probs.npy")
preds = (probs >= 0.5).astype(int)

df["y_true"] = y_true
df["pred"] = preds
df["is_error"] = (df["y_true"] != df["pred"])

n_errors = df["is_error"].sum()
n_correct = (~df["is_error"]).sum()
print(f"Total test reviews: {len(df)}")
print(f"Errors: {n_errors} ({n_errors/len(df)*100:.2f}%)")
print(f"Correct: {n_correct} ({n_correct/len(df)*100:.2f}%)")

# 2. Rule regex definitions
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

# Evaluate on entire dataset
cats = []
labels = []
for idx, row in df.iterrows():
    c_id, c_lbl = assign_category(row)
    cats.append(c_id)
    labels.append(c_lbl)

df["category_id"] = cats
df["category_label"] = labels

print("\n" + "=" * 80)
print(f"{'Category ID':<22} | {'Errors (n=1,901)':<18} | {'Correct (n=70,945)':<20} | {'Error Rate':<12}")
print("=" * 80)

cat_order = ["rating_inversion", "mixed_clauses", "packaging_defect", "sarcasm", "sequence_truncation", "domain_lexicon", "indirect_negation"]

for cid in cat_order:
    sub = df[df["category_id"] == cid]
    err_count = sub["is_error"].sum()
    corr_count = (~sub["is_error"]).sum()
    total_cat = len(sub)
    
    err_pct = err_count / n_errors * 100
    corr_pct = corr_count / n_correct * 100
    cat_err_rate = err_count / total_cat * 100
    
    lbl = sub.iloc[0]["category_label"]
    print(f"{cid:<22} | {err_count:>5} ({err_pct:>5.1f}%)      | {corr_count:>6} ({corr_pct:>5.1f}%)        | {cat_err_rate:>5.2f}%")

print("=" * 80)

# Save the full classified reviews dataframe for reproducible audit
output_path = ROOT / "reports" / "taxonomy_full_audit.csv"
df[["text", "y_true", "pred", "is_error", "category_id", "category_label"]].to_csv(output_path, index=True)
print(f"\nSaved full per-review classification audit to: {output_path}")
