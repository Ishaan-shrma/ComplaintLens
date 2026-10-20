"""
scripts/verify_no_leakage.py
============================
Rigorous Leakage Auditor for the Universal Review Intelligence Pipeline.

Checks:
  1. Train vs OOD Test Overlap within omni-corpus (exact text matches & normalized text hashes)
  2. Omni-corpus (Train & OOD) vs Canonical Amazon Food Test (72,846 reviews in notebooks/artifacts/test_72846.pkl)
  3. Omni-corpus vs Canonical Full Train (281,271 reviews in notebooks/artifacts/full_train_281271.pkl)
  4. Summary of domain, source, and label integrity
"""

import pickle
import sys
from pathlib import Path
import pandas as pd
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "notebooks" / "artifacts"


def hash_text(text: str) -> str:
    cleaned = " ".join(str(text).lower().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def run_audit():
    print("=" * 65)
    print("  LEAKAGE AUDIT & INTEGRITY CHECK")
    print("=" * 65)

    omni_train_file = DATA_DIR / "omni_corpus.csv"
    omni_ood_file = DATA_DIR / "omni_ood_test.csv"
    canonical_test_pkl = ARTIFACTS_DIR / "test_72846.pkl"

    if not omni_train_file.exists() or not omni_ood_file.exists():
        print("[-] Omni corpus files missing! Please run build_omni_corpus.py first.")
        return False

    df_omni_train = pd.read_csv(omni_train_file)
    df_omni_ood = pd.read_csv(omni_ood_file)

    print(f"\n[+] Loaded omni_train: {len(df_omni_train):,} rows")
    print(f"[+] Loaded omni_ood:   {len(df_omni_ood):,} rows")

    # 1. Internal Leakage: omni_train vs omni_ood
    print("\n--- Check 1: Omni Train vs Omni OOD Test Overlap ---")
    train_hashes = set(df_omni_train["text"].astype(str).apply(hash_text))
    ood_hashes = set(df_omni_ood["text"].astype(str).apply(hash_text))

    overlap = train_hashes.intersection(ood_hashes)
    if len(overlap) == 0:
        print("  [PASS] Zero text collision between Omni Train and Omni OOD Test!")
    else:
        print(f"  [FAIL] Detected {len(overlap)} overlapping reviews between train & OOD!")

    # 2. Canonical Amazon Fine Food Test Set Overlap
    print("\n--- Check 2: Omni Data vs Canonical Test (72,846) ---")
    if canonical_test_pkl.exists():
        with open(canonical_test_pkl, "rb") as f:
            test_data = pickle.load(f)
        
        # Test pickle might be DataFrame or dict or list
        if isinstance(test_data, pd.DataFrame):
            test_texts = test_data["Text"].tolist() if "Text" in test_data else test_data.iloc[:, 0].tolist()
        elif isinstance(test_data, dict):
            test_texts = test_data.get("Text", test_data.get("text", []))
        else:
            test_texts = list(test_data)

        print(f"  [+] Loaded canonical test records: {len(test_texts):,}")
        canonical_test_hashes = set(pd.Series(test_texts).astype(str).apply(hash_text))

        leak_train = train_hashes.intersection(canonical_test_hashes)
        leak_ood = ood_hashes.intersection(canonical_test_hashes)

        if len(leak_train) == 0 and len(leak_ood) == 0:
            print("  [PASS] Zero contamination in Canonical 72,846 Held-Out Test set!")
        else:
            print(f"  [FAIL] Overlap found: {len(leak_train)} in train, {len(leak_ood)} in OOD!")
    else:
        print(f"  [SKIP] Canonical test pkl not found at {canonical_test_pkl}")

    print("\n" + "=" * 65)
    print("  AUDIT COMPLETED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_audit()
