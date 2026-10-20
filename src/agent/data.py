"""Cached, read-only loaders. Nothing in this package writes to disk."""
from functools import lru_cache

import pandas as pd

from .config import FAISS_INDEX, FAISS_UNIVERSAL_INDEX, EMBED_MODEL, REVIEWS_CSV, SAMPLE_META_CSV, UNIVERSAL_META_CSV

_WANTED = ["Id", "ProductId", "Score", "sentiment_label", "review_year", "review_month",
           "helpfulness_ratio", "datetime"]


@lru_cache(maxsize=1)
def reviews() -> pd.DataFrame:
    """Structured columns of the cleaned dataset (no review text), for KPIs."""
    header = pd.read_csv(REVIEWS_CSV, nrows=0).columns
    need = {"Id", "Score", "sentiment_label"}
    if need - set(header):
        raise KeyError(f"reviews_cleaned.csv is missing columns: {sorted(need - set(header))}")
    return pd.read_csv(REVIEWS_CSV, usecols=[c for c in _WANTED if c in header])


@lru_cache(maxsize=1)
def sample_meta() -> pd.DataFrame:
    """Metadata of the indexed sample, indexed by review Id, with the review date joined in."""
    target_meta = UNIVERSAL_META_CSV if UNIVERSAL_META_CSV.exists() else SAMPLE_META_CSV
    meta = pd.read_csv(target_meta)
    if not meta["Id"].is_unique:
        raise ValueError(f"{target_meta.name} has duplicate Ids")
    extra = [c for c in ("Id", "review_year", "datetime") if c in reviews().columns]
    return meta.merge(reviews()[extra], on="Id", how="left").set_index("Id")


@lru_cache(maxsize=1)
def faiss_index():
    import faiss
    target_idx = FAISS_UNIVERSAL_INDEX if FAISS_UNIVERSAL_INDEX.exists() else FAISS_INDEX
    return faiss.read_index(str(target_idx))


@lru_cache(maxsize=1)
def embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
