import json
from pathlib import Path
import pandas as pd

ROOT = Path("A:/DS-Review-AI")

# 1. Inspect omni_corpus.csv
omni_csv = ROOT / "data" / "processed" / "omni_corpus.csv"
if omni_csv.exists():
    df = pd.read_csv(omni_csv)
    print("=== OMNI CORPUS SUMMARY ===")
    print(f"Total rows: {len(df)}")
    print(df['source'].value_counts())
    print("\nCounts by source & sentiment_label:")
    print(df.groupby(['source', 'sentiment_label']).size())

# 2. Inspect how train_deberta.py sampled 42,190
omni_filtered = df[df["sentiment_label"].isin(["negative", "positive"])].copy()
print(f"\nTotal binary filtered rows across all 3 sources: {len(omni_filtered)}")
print(omni_filtered['source'].value_counts())

SEED = 42
n_omni = int(281271 * 0.15)
omni_sample = omni_filtered.sample(min(n_omni, len(omni_filtered)), random_state=SEED)
print(f"\n=== REPLAY SAMPLE ({len(omni_sample)} rows) PER PLATFORM BREAKDOWN ===")
print(omni_sample['source'].value_counts())
print("\nReplay sample by source & label:")
print(omni_sample.groupby(['source', 'sentiment_label']).size())

# 3. Check remaining rows (disjoint test sets)
remaining = omni_filtered.drop(omni_sample.index)
print(f"\n=== REMAINING (HELD-OUT DISJOINT) ROWS ({len(remaining)} rows) ===")
print(remaining['source'].value_counts())
print("\nRemaining by source & label:")
print(remaining.groupby(['source', 'sentiment_label']).size())
