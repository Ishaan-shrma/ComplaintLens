"""Cached, read-only loaders. Nothing in this package writes to disk."""
from functools import lru_cache
import os
import pandas as pd

from .config import (
    FAISS_INDEX,
    FAISS_UNIVERSAL_INDEX,
    EMBED_MODEL,
    REVIEWS_CSV,
    SAMPLE_REVIEWS_CSV,
    ACTIVE_REVIEWS_CSV,
    DATA_MODE,
    SAMPLE_META_CSV,
    UNIVERSAL_META_CSV,
)

_WANTED = ["Id", "ProductId", "Score", "sentiment_label", "review_year", "review_month",
           "helpfulness_ratio", "datetime"]


@lru_cache(maxsize=1)
def reviews() -> pd.DataFrame:
    """Structured columns of the cleaned dataset (no review text), for KPIs."""
    target_csv = ACTIVE_REVIEWS_CSV
    if not target_csv.exists():
        raise FileNotFoundError(f"Neither full reviews_cleaned.csv nor sample_reviews_5k.csv found at {target_csv}")

    header = pd.read_csv(target_csv, nrows=0).columns
    need = {"Id", "Score"}
    if need - set(header):
        raise KeyError(f"{target_csv.name} is missing columns: {sorted(need - set(header))}")

    cols = [c for c in _WANTED if c in header]
    df = pd.read_csv(target_csv, usecols=cols)

    # Derive missing columns if running on sample_reviews_5k.csv
    if "sentiment_label" not in df.columns and "Score" in df.columns:
        df["sentiment_label"] = df["Score"].map({1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})
    if "ProductId" not in df.columns:
        df["ProductId"] = "SAMPLE_PROD"
    if "datetime" not in df.columns and "Time" in header:
        time_series = pd.read_csv(target_csv, usecols=["Time"])["Time"]
        dt = pd.to_datetime(time_series, unit="s")
        df["datetime"] = dt.dt.strftime("%Y-%m-%d")
        df["review_year"] = dt.dt.year
        df["review_month"] = dt.dt.month
    if "helpfulness_ratio" not in df.columns:
        df["helpfulness_ratio"] = 1.0

    return df


@lru_cache(maxsize=1)
def sample_meta() -> pd.DataFrame:
    """Metadata of the indexed sample, indexed by review Id, with the review date joined in."""
    target_meta = UNIVERSAL_META_CSV if UNIVERSAL_META_CSV.exists() else SAMPLE_META_CSV
    if target_meta.exists():
        meta = pd.read_csv(target_meta)
        if not meta["Id"].is_unique:
            raise ValueError(f"{target_meta.name} has duplicate Ids")
        extra = [c for c in ("Id", "review_year", "datetime") if c in reviews().columns]
        return meta.merge(reviews()[extra], on="Id", how="left").set_index("Id")
    elif SAMPLE_REVIEWS_CSV.exists():
        # Fallback to local 5k sample metadata
        meta = pd.read_csv(SAMPLE_REVIEWS_CSV)
        if "sentiment_label" not in meta.columns and "Score" in meta.columns:
            meta["sentiment_label"] = meta["Score"].map({1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})
        if "ProductId" not in meta.columns:
            meta["ProductId"] = "SAMPLE_PROD"
        if "Time" in meta.columns:
            dt = pd.to_datetime(meta["Time"], unit="s")
            meta["datetime"] = dt.dt.strftime("%Y-%m-%d")
            meta["review_year"] = dt.dt.year
        return meta.set_index("Id")
    else:
        raise FileNotFoundError("No sample metadata or sample reviews found.")


@lru_cache(maxsize=1)
def faiss_index():
    import faiss
    target_idx = FAISS_UNIVERSAL_INDEX if FAISS_UNIVERSAL_INDEX.exists() else FAISS_INDEX
    if target_idx.exists():
        return faiss.read_index(str(target_idx))

    # Graceful fallback: If index is not downloaded, build lightweight in-memory index from sample
    meta = sample_meta()
    texts = meta["Text"].fillna("").astype(str).tolist()[:500]
    emb = embedder().encode(texts, normalize_embeddings=True).astype("float32")
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


@lru_cache(maxsize=1)
def embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
