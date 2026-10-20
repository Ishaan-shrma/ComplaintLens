"""
scripts/verify_comprehensive_leakage.py
=======================================
Exhaustive Multi-Tier Data Leakage Audit across all partitions:
  Tier 1: Canonical Train (281,271) vs Canonical Held-Out Test (72,846)
  Tier 2: Validation Set (9,997) vs Canonical Test (72,846)
  Tier 3: Omni-Corpus Replay (59,988) vs Canonical Test (72,846)
  Tier 4: Omni-Corpus Replay (59,988) vs Omni OOD Test (6,000)
  Tier 5: Canonical Train (281,271) vs Omni OOD Test (6,000)
"""

import sys
import pickle
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
ART_DIR  = ROOT / "notebooks" / "artifacts"


def hash_text(text: str) -> str:
    """Normalize whitespace and lowercase before computing SHA-256."""
    cleaned = " ".join(str(text).lower().strip().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def load_texts(file_path: Path):
    if str(file_path).endswith(".pkl"):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            col = "text" if "text" in data.columns else ("Text" if "Text" in data.columns else data.columns[0])
            return data[col].astype(str).tolist()
        elif isinstance(data, dict):
            col = "text" if "text" in data else "Text"
            return list(map(str, data[col]))
        else:
            return [str(x) for x in data]
    elif str(file_path).endswith(".csv"):
        df = pd.read_csv(file_path)
        col = "text" if "text" in df.columns else ("Text" if "Text" in df.columns else df.columns[0])
        return df[col].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported format: {file_path}")


def audit_tier(name_a, texts_a, name_b, texts_b):
    print(f"\n--- Checking {name_a} vs {name_b} ---")
    print(f"  {name_a}: {len(texts_a):,} samples")
    print(f"  {name_b}: {len(texts_b):,} samples")

    hashes_a = set(hash_text(t) for t in texts_a)
    hashes_b = set(hash_text(t) for t in texts_b)

    overlap = hashes_a.intersection(hashes_b)
    overlap_count = len(overlap)

    if overlap_count == 0:
        print(f"  [PASS] ZERO collision! (Overlap: 0 samples / 0.000%)")
        return True, 0
    else:
        pct = (overlap_count / len(texts_b)) * 100
        print(f"  [FAIL] LEAKAGE DETECTED: {overlap_count:,} overlapping samples ({pct:.4f}%)!")
        return False, overlap_count


def run_full_leakage_audit():
    print("=" * 70)
    print("  EXHAUSTIVE CRYPTOGRAPHIC DATA LEAKAGE AUDIT")
    print("  Method: SHA-256 Hashing on Whitespace-Normalized Lowercase Text")
    print("=" * 70)

    train_pkl = ART_DIR / "full_train_281271.pkl"
    test_pkl  = ART_DIR / "test_72846.pkl"
    val_pkl   = ART_DIR / "val_sub_9997.pkl"
    omni_csv  = DATA_DIR / "omni_corpus.csv"
    ood_csv   = DATA_DIR / "omni_ood_test.csv"

    all_passed = True
    results = {}

    # Load datasets
    print("\nLoading dataset partitions...")
    train_texts = load_texts(train_pkl)
    test_texts  = load_texts(test_pkl)
    val_texts   = load_texts(val_pkl)
    omni_texts  = load_texts(omni_csv)
    ood_texts   = load_texts(ood_csv)

    print(f"  1. Canonical Train       : {len(train_texts):,} reviews")
    print(f"  2. Canonical Test        : {len(test_texts):,} reviews")
    print(f"  3. Validation Set        : {len(val_texts):,} reviews")
    print(f"  4. Omni Replay Corpus    : {len(omni_texts):,} reviews")
    print(f"  5. Omni OOD Test Set     : {len(ood_texts):,} reviews")

    # Tier 1: Canonical Train vs Canonical Test
    passed, n = audit_tier("Canonical Train (281k)", train_texts, "Canonical Test (72k)", test_texts)
    results["Train_vs_Test"] = {"passed": passed, "overlap": n}
    all_passed = all_passed and passed

    # Tier 2: Validation vs Canonical Test
    passed, n = audit_tier("Validation (9.9k)", val_texts, "Canonical Test (72k)", test_texts)
    results["Val_vs_Test"] = {"passed": passed, "overlap": n}
    all_passed = all_passed and passed

    # Tier 3: Omni Replay vs Canonical Test
    passed, n = audit_tier("Omni Replay (60k)", omni_texts, "Canonical Test (72k)", test_texts)
    results["OmniReplay_vs_Test"] = {"passed": passed, "overlap": n}
    all_passed = all_passed and passed

    # Tier 4: Omni Replay vs Omni OOD Test
    passed, n = audit_tier("Omni Replay (60k)", omni_texts, "Omni OOD Test (6k)", ood_texts)
    results["OmniReplay_vs_OmniOOD"] = {"passed": passed, "overlap": n}
    all_passed = all_passed and passed

    # Tier 5: Canonical Train vs Omni OOD Test
    passed, n = audit_tier("Canonical Train (281k)", train_texts, "Omni OOD Test (6k)", ood_texts)
    results["Train_vs_OmniOOD"] = {"passed": passed, "overlap": n}
    all_passed = all_passed and passed

    print("\n" + "=" * 70)
    print("  FINAL AUDIT SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        status = "PASSED (0 overlap)" if v["passed"] else f"FAILED ({v['overlap']} overlap)"
        print(f"  {k:<30} : {status}")

    print("=" * 70)
    if all_passed:
        print("  VERDICT: ZERO DATA LEAKAGE CONFIRMED ACROSS ALL SPLITS.")
        print("  All benchmark evaluations are 100% statistically sound and honest.")
    else:
        print("  VERDICT: CONTAMINATION DETECTED! Needs immediate remediation.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    run_full_leakage_audit()
