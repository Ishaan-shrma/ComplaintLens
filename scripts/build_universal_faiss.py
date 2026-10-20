"""
Universal FAISS Index Builder (160,000 Multi-Platform Reviews)
DS-Review-AI Universal Intelligence Platform

Builds a unified 159,988-vector semantic search index by combining:
1. Canonical Amazon Fine Food sample (100,000 vectors)
2. Omni-Corpus Replay Buffer (59,988 vectors from Flipkart, Yelp, and IMDb)

Output artifacts:
- models/faiss_universal_160k.bin
- data/processed/universal_sample_metadata.csv
"""
import sys
import time
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512

def extract_summary(text: str, max_chars: int = 60) -> str:
    """Extracts a concise headline from review text."""
    if not isinstance(text, str) or not text.strip():
        return "Review"
    # Take first sentence or first max_chars
    first_period = text.find(".")
    if 10 < first_period <= max_chars:
        return text[:first_period].strip()
    return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")

def main():
    print("=" * 70)
    print("  [*] BUILDING UNIVERSAL 160K FAISS VECTOR DATABASE")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Existing Amazon Baseline FAISS Index & Metadata
    base_index_path = PROJECT_ROOT / "models" / "faiss_index_with_ids.bin"
    base_meta_path = PROJECT_ROOT / "data" / "processed" / "embedding_sample_metadata.csv"
    
    if not base_index_path.exists() or not base_meta_path.exists():
        raise FileNotFoundError("Baseline Amazon FAISS artifacts not found!")

    print(f"\n[1/5] Loading baseline Amazon FAISS index from {base_index_path.name}...")
    base_index = faiss.read_index(str(base_index_path))
    print(f"      Loaded {base_index.ntotal:,} Amazon vectors.")

    print(f"[2/5] Loading baseline Amazon metadata from {base_meta_path.name}...")
    df_amazon = pd.read_csv(base_meta_path)
    df_amazon["platform"] = "amazon"
    df_amazon["domain"] = "fine_food"
    print(f"      Amazon metadata shape: {df_amazon.shape}")

    # 2. Load Omni-Corpus (Flipkart, Yelp, IMDb)
    omni_path = PROJECT_ROOT / "data" / "processed" / "omni_corpus.csv"
    if not omni_path.exists():
        raise FileNotFoundError(f"Omni corpus not found at {omni_path}!")

    print(f"\n[3/5] Loading Omni-Corpus from {omni_path.name}...")
    df_omni = pd.read_csv(omni_path)
    n_omni = len(df_omni)
    print(f"      Loaded {n_omni:,} Omni-Corpus reviews across:")
    for src, cnt in df_omni["source"].value_counts().items():
        print(f"        - {src.capitalize():<12}: {cnt:,} reviews")

    # Format Omni metadata with non-colliding IDs (starting at 1,000,000)
    omni_ids = np.arange(1_000_000, 1_000_000 + n_omni, dtype=np.int64)
    
    score_map = {"positive": 5, "neutral": 3, "negative": 1}
    omni_scores = [score_map.get(s.lower(), 3) for s in df_omni["sentiment_label"]]
    omni_product_ids = [f"{src.upper()}_{i:06d}" for i, src in enumerate(df_omni["source"])]
    omni_summaries = [extract_summary(t) for t in df_omni["text"]]

    df_omni_formatted = pd.DataFrame({
        "Id": omni_ids,
        "ProductId": omni_product_ids,
        "Score": omni_scores,
        "sentiment_label": df_omni["sentiment_label"].str.lower().str.strip(),
        "Summary": omni_summaries,
        "Text": df_omni["text"],
        "platform": df_omni["source"].str.lower().str.strip(),
        "domain": df_omni["domain"]
    })

    # 3. Embed Omni-Corpus Texts via all-MiniLM-L6-v2 on GPU
    print(f"\n[4/5] Embedding {n_omni:,} Omni reviews with '{EMBED_MODEL}'...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    
    t0 = time.time()
    omni_texts = df_omni["text"].tolist()
    omni_embeddings = embedder.encode(
        omni_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    embed_time = time.time() - t0
    rate = n_omni / embed_time if embed_time > 0 else 0
    print(f"      Completed in {embed_time:.2f}s ({rate:.0f} texts/sec).")
    print(f"      Embeddings shape: {omni_embeddings.shape}")

    # 4. Append Omni Vectors to FAISS Index
    print(f"\n[5/5] Appending Omni vectors to FAISS index...")
    base_index.add_with_ids(omni_embeddings, omni_ids)
    total_vectors = base_index.ntotal
    print(f"      Unified FAISS Index Vector Count: {total_vectors:,}")

    # 5. Save Artifacts
    out_index_path = PROJECT_ROOT / "models" / "faiss_universal_160k.bin"
    out_meta_path = PROJECT_ROOT / "data" / "processed" / "universal_sample_metadata.csv"

    print(f"\n[*] Writing unified FAISS index to {out_index_path}...")
    faiss.write_index(base_index, str(out_index_path))
    index_size_mb = out_index_path.stat().st_size / (1024 * 1024)
    print(f"      Index saved successfully ({index_size_mb:.1f} MB).")

    print(f"[*] Writing unified metadata to {out_meta_path}...")
    df_universal = pd.concat([df_amazon, df_omni_formatted], ignore_index=True)
    df_universal.to_csv(out_meta_path, index=False)
    meta_size_mb = out_meta_path.stat().st_size / (1024 * 1024)
    print(f"      Metadata saved successfully ({df_universal.shape[0]:,} rows, {meta_size_mb:.1f} MB).")

    # 6. Sanity Test Query
    print("\n" + "=" * 70)
    print("  VERIFICATION TEST: CROSS-DOMAIN RETRIEVAL")
    print("=" * 70)
    
    test_queries = [
        ("delicious organic vanilla bean black tea", "amazon"),
        ("phone battery drains rapidly overnight and overheats", "flipkart"),
        ("authentic wood fired pizza and creamy carbonara pasta", "yelp"),
        ("dull plot terrible acting and wasted potential screenplay", "imdb")
    ]
    
    for q, expected_domain in test_queries:
        q_vec = embedder.encode([q], normalize_embeddings=True).astype("float32")
        scores, ids = base_index.search(q_vec, 3)
        print(f"\nQuery: '{q}' (Target: {expected_domain})")
        for rank, (score, r_id) in enumerate(zip(scores[0], ids[0]), 1):
            match_row = df_universal[df_universal["Id"] == r_id].iloc[0]
            print(f"  #{rank} [Score: {score:.4f}] [{match_row['platform'].upper()} {match_row['Score']}★]: {match_row['Summary'][:75]}")

    print("\n[SUCCESS] UNIVERSAL 160K FAISS DATABASE BUILD COMPLETE!")

if __name__ == "__main__":
    main()
