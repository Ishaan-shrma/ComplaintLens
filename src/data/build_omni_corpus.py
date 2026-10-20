"""
src/data/build_omni_corpus.py
==============================
Builds the 60,000-review multi-domain experience replay buffer used to make
the model robust to domain shift, Hinglish, slang, emojis, and sarcasm.

Sources (all from HuggingFace datasets — no manual download needed):
  1. Yelp Review Full       (stanfordnlp/yelp_review_full)         → 15k
  2. IMDb                   (stanfordnlp/imdb)                     → 15k
  3. Amazon Multi-Category  (McAuley-Lab/Amazon-Reviews-2023)      → 15k
     sub-datasets: Electronics, Software, Kindle, Clothing
  4. Flipkart               (local CSV fallback or Kaggle download) → 15k

Sentiment mapping (all sources → 3-class):
  Positive  (rating ≥ 4, or label == 1 for IMDb)
  Negative  (rating ≤ 2, or label == 0 for IMDb)
  Neutral   (rating == 3)  — IMDb has no neutral; skipped for IMDb.

Output:
  data/processed/omni_corpus.csv   — columns: text, sentiment_label, source, domain
  data/processed/omni_ood_test.csv — 2,000 held-out OOD reviews per source (8k total)

Usage (from project root, inside .venv):
  python src/data/build_omni_corpus.py

Flags:
  --n_per_source   Reviews per source for training buffer  (default 15000)
  --ood_per_source OOD test reviews per source             (default 2000)
  --seed           Random seed                             (default 42)
  --output_dir     Output directory                        (default data/processed/)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data.normalizer import normalize

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_N        = 15_000
DEFAULT_OOD      = 2_000
DEFAULT_SEED     = 42
OUTPUT_DIR       = PROJECT_ROOT / "data" / "processed"
FLIPKART_FALLBACK = PROJECT_ROOT / "data" / "raw" / "flipkart_reviews.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rating_to_label(rating: float) -> str | None:
    """Map a numeric star rating to a 3-class sentiment label. None = discard."""
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    return None


def _sample_balanced(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows from df with approximate class balance.
    If a class has fewer than n//3 samples, take all of them and compensate
    from other classes so total stays close to n.
    """
    classes = df["sentiment_label"].unique().tolist()
    per_class = n // len(classes)
    parts = []
    for cls in classes:
        subset = df[df["sentiment_label"] == cls]
        take = min(len(subset), per_class)
        parts.append(subset.sample(take, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    # Top-up if under n due to class imbalance
    if len(sampled) < n:
        remainder = df.drop(sampled.index, errors="ignore")
        top_up = remainder.sample(min(n - len(sampled), len(remainder)),
                                  random_state=seed)
        sampled = pd.concat([sampled, top_up], ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def _split_train_ood(df: pd.DataFrame, n_train: int, n_ood: int,
                     seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a source dataframe into train and OOD test portions."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    ood   = df.iloc[:n_ood].copy()
    train = df.iloc[n_ood:n_ood + n_train].copy()
    return train, ood


# ── Source loaders ────────────────────────────────────────────────────────────

def load_yelp(n: int, ood: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from stanfordnlp/yelp_review_full via HuggingFace datasets."""
    print("  [Yelp] Loading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    df = ds.to_pandas()[["text", "label"]].copy()
    # Yelp labels: 0=1-star ... 4=5-star
    df["rating"] = df["label"] + 1
    df["sentiment_label"] = df["rating"].map(_rating_to_label)
    df = df.dropna(subset=["sentiment_label"])
    df = df.sample(min(n + ood, len(df)), random_state=seed).reset_index(drop=True)
    df = df.rename(columns={"text": "raw_text"})
    df["text"]   = df["raw_text"].apply(normalize)
    df["source"] = "yelp"
    df["domain"] = "restaurant_local"
    df = df[["text", "sentiment_label", "source", "domain"]]
    train, ood_df = _split_train_ood(df, n, ood, seed)
    print(f"  [Yelp] train={len(train):,}  ood={len(ood_df):,}  "
          f"labels={train['sentiment_label'].value_counts().to_dict()}")
    return train, ood_df


def load_imdb(n: int, ood: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from stanfordnlp/imdb via HuggingFace datasets."""
    print("  [IMDb] Loading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train")
    df = ds.to_pandas()[["text", "label"]].copy()
    # IMDb labels: 0=negative, 1=positive (no neutral)
    df["sentiment_label"] = df["label"].map({0: "negative", 1: "positive"})
    df = df.dropna(subset=["sentiment_label"])
    df = df.sample(min(n + ood, len(df)), random_state=seed).reset_index(drop=True)
    df = df.rename(columns={"text": "raw_text"})
    df["text"]   = df["raw_text"].apply(normalize)
    df["source"] = "imdb"
    df["domain"] = "movies_entertainment"
    df = df[["text", "sentiment_label", "source", "domain"]]
    train, ood_df = _split_train_ood(df, n, ood, seed)
    print(f"  [IMDb] train={len(train):,}  ood={len(ood_df):,}  "
          f"labels={train['sentiment_label'].value_counts().to_dict()}")
    return train, ood_df


def load_amazon_multi(n: int, ood: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Amazon Reviews 2023 across 4 sub-categories.
    Uses McAuley-Lab/Amazon-Reviews-2023 with the 5core (deduplicated) split.
    """
    print("  [Amazon Multi] Loading from HuggingFace...")
    from datasets import load_dataset

    SUB_CATEGORIES = [
        ("Electronics",      "Electronics_5core"),
        ("Software",         "Software_5core"),
        ("Kindle",           "Kindle_Store_5core"),
        ("Clothing_Apparel", "Clothing_Shoes_and_Jewelry_5core"),
    ]

    per_cat = (n + ood) // len(SUB_CATEGORIES) + 500  # slight oversample
    parts = []
    for domain, sub in SUB_CATEGORIES:
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                sub,
                split="full",
            )
            cat_df = ds.to_pandas()
            # Column name varies; try "overall" then "rating"
            rating_col = "overall" if "overall" in cat_df.columns else "rating"
            text_col   = "reviewText" if "reviewText" in cat_df.columns else "text"
            cat_df = cat_df[[text_col, rating_col]].dropna()
            cat_df = cat_df.rename(columns={text_col: "raw_text", rating_col: "rating"})
            cat_df["sentiment_label"] = cat_df["rating"].apply(_rating_to_label)
            cat_df = cat_df.dropna(subset=["sentiment_label"])
            cat_df = cat_df.sample(min(per_cat, len(cat_df)), random_state=seed)
            cat_df["text"]   = cat_df["raw_text"].apply(normalize)
            cat_df["source"] = "amazon_multi"
            cat_df["domain"] = domain.lower()
            parts.append(cat_df[["text", "sentiment_label", "source", "domain"]])
            print(f"    [{domain}] loaded {len(cat_df):,} rows")
        except Exception as e:
            print(f"    [{domain}] FAILED: {e} — skipping")

    if not parts:
        print("  [Amazon Multi] All sub-categories failed — returning empty DataFrames")
        empty = pd.DataFrame(columns=["text", "sentiment_label", "source", "domain"])
        return empty, empty

    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    train, ood_df = _split_train_ood(df, n, ood, seed)
    print(f"  [Amazon Multi] train={len(train):,}  ood={len(ood_df):,}  "
          f"labels={train['sentiment_label'].value_counts().to_dict()}")
    return train, ood_df


def load_flipkart(n: int, ood: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Flipkart reviews.
    Priority:
      1. Local CSV at data/raw/flipkart_reviews.csv (if manually downloaded)
      2. HuggingFace: it802/flipkart-product-reviews
      3. Synthetic fallback — generates labelled examples from templates so the
         pipeline never fails completely on a machine without internet / Kaggle.
    """
    print("  [Flipkart] Loading...")

    # ── Option 1: local file ─────────────────────────────────────────────────
    if FLIPKART_FALLBACK.exists():
        print(f"    Using local file: {FLIPKART_FALLBACK}")
        df = pd.read_csv(FLIPKART_FALLBACK)
        # Common column names in Kaggle Flipkart CSVs
        text_col   = next((c for c in df.columns if "review" in c.lower() or "text" in c.lower()), None)
        rating_col = next((c for c in df.columns if "rating" in c.lower() or "score" in c.lower()), None)
        if text_col and rating_col:
            df = df[[text_col, rating_col]].dropna()
            df = df.rename(columns={text_col: "raw_text", rating_col: "rating"})
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df["sentiment_label"] = df["rating"].apply(_rating_to_label)
            df = df.dropna(subset=["sentiment_label"])
        else:
            print("    Could not identify text/rating columns — trying HuggingFace")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # ── Option 2: HuggingFace ────────────────────────────────────────────────
    if df.empty:
        try:
            from datasets import load_dataset
            ds = load_dataset("it802/flipkart-product-reviews",
                              split="train")
            df = ds.to_pandas()
            text_col   = next((c for c in df.columns if "review" in c.lower()), None)
            rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
            if text_col and rating_col:
                df = df[[text_col, rating_col]].dropna()
                df = df.rename(columns={text_col: "raw_text", rating_col: "rating"})
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
                df["sentiment_label"] = df["rating"].apply(_rating_to_label)
                df = df.dropna(subset=["sentiment_label"])
            else:
                df = pd.DataFrame()
        except Exception as e:
            print(f"    HuggingFace load failed: {e}")
            df = pd.DataFrame()

    # ── Option 3: Template-based synthetic fallback ───────────────────────────
    if df.empty:
        print("    Generating synthetic Flipkart fallback data...")
        df = _generate_synthetic_flipkart(n + ood, seed)

    if not df.empty:
        df = df.sample(min(n + ood, len(df)), random_state=seed).reset_index(drop=True)
        df["text"]   = df["raw_text"].apply(normalize)
        df["source"] = "flipkart"
        df["domain"] = "indian_ecommerce"
        df = df[["text", "sentiment_label", "source", "domain"]]

    train, ood_df = _split_train_ood(df, n, ood, seed)
    print(f"  [Flipkart] train={len(train):,}  ood={len(ood_df):,}  "
          f"labels={train['sentiment_label'].value_counts().to_dict()}")
    return train, ood_df


def _generate_synthetic_flipkart(n: int, seed: int) -> pd.DataFrame:
    """
    Synthesize diverse Indian e-commerce / Hinglish reviews using combinatoric
    patterns with dynamic vocabulary so all generated samples are unique.
    """
    rng = np.random.default_rng(seed)

    subjects = [
        "Product", "Item", "Quality", "Packaging", "Build quality",
        "Delivery", "Performance", "Sound quality", "Battery life", "Camera"
    ]
    pos_adjs = [
        "ekdum mast", "paisa vasool", "zabardast", "badhiya", "bilkul sahi",
        "top notch", "fantastic", "kadak", "superb", "first class"
    ]
    pos_actions = [
        "loved it completely", "will order again definitely", "works smoothly",
        "arrived very fast", "highly recommended to everyone", "worth every single rupee"
    ]

    neg_adjs = [
        "ekdum bakwas", "bekar quality", "ghatiya", "faltu", "total waste",
        "bilkul useless", "nakli piece", "not working", "defective"
    ]
    neg_actions = [
        "money wasted", "requested return immediately", "stopped working in 2 days",
        "dhoka hai seller fraud", "arrived damaged and broken", "never buying from here again"
    ]

    neut_adjs = [
        "okay okay", "theek thaak", "average", "decent for price", "just normal", "satisfactory"
    ]
    neut_actions = [
        "does the basic job", "nothing special", "expected better but acceptable", "average experience"
    ]

    emojis_pos = ["", " [fire]", " [perfect]", " [thumbs_up]", " [star]"]
    emojis_neg = ["", " [trash]", " [dead]", " [thumbs_down]", " [crying]"]

    pos_n = int(n * 0.60)
    neg_n = int(n * 0.25)
    neut_n = n - pos_n - neg_n

    rows = []
    seen = set()

    def make_unique(gen_func):
        for _ in range(100):
            t = gen_func()
            if t not in seen:
                seen.add(t)
                return t
        return gen_func() + f" (ref #{rng.integers(10000, 99999)})"

    for _ in range(pos_n):
        txt = make_unique(lambda: f"{rng.choice(subjects)} is {rng.choice(pos_adjs)}, {rng.choice(pos_actions)}{rng.choice(emojis_pos)}.")
        rows.append({"raw_text": txt, "sentiment_label": "positive"})

    for _ in range(neg_n):
        txt = make_unique(lambda: f"{rng.choice(subjects)} is {rng.choice(neg_adjs)}, {rng.choice(neg_actions)}{rng.choice(emojis_neg)}.")
        rows.append({"raw_text": txt, "sentiment_label": "negative"})

    for _ in range(neut_n):
        txt = make_unique(lambda: f"{rng.choice(subjects)} is {rng.choice(neut_adjs)}, {rng.choice(neut_actions)}.")
        rows.append({"raw_text": txt, "sentiment_label": "neutral"})

    return pd.DataFrame(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_corpus(n_per_source: int, ood_per_source: int, seed: int,
                 output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parts = []
    ood_parts   = []

    loaders = [
        ("Yelp",     load_yelp,     25_000, 2_000),
        ("IMDb",     load_imdb,     20_000, 2_000),
        ("Flipkart", load_flipkart, 15_000, 2_000),
    ]

    for name, loader_fn, n_tr, n_od in loaders:
        print(f"\n[{name}] Target: {n_tr:,} train, {n_od:,} OOD...")
        try:
            tr, od = loader_fn(n_tr, n_od, seed)
            if not tr.empty:
                train_parts.append(tr)
            if not od.empty:
                ood_parts.append(od)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    ood_df   = pd.concat(ood_parts,   ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)

    train_path = output_dir / "omni_corpus.csv"
    ood_path   = output_dir / "omni_ood_test.csv"

    train_df.to_csv(train_path, index=False)
    ood_df.to_csv(ood_path,   index=False)

    print("\n" + "=" * 60)
    print("  Omni-Corpus Build Complete")
    print("=" * 60)
    print(f"  Training buffer : {len(train_df):,} reviews  →  {train_path}")
    print(f"  OOD test set    : {len(ood_df):,} reviews  →  {ood_path}")
    print(f"\n  Training label distribution:")
    for lbl, cnt in train_df["sentiment_label"].value_counts().items():
        print(f"    {lbl:<12} {cnt:>7,}  ({cnt/len(train_df)*100:.1f}%)")
    print(f"\n  Source breakdown (training):")
    for src, cnt in train_df["source"].value_counts().items():
        print(f"    {src:<20} {cnt:>7,}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Build Omni-Domain Corpus")
    p.add_argument("--n_per_source",   type=int, default=DEFAULT_N,
                   help=f"Reviews per source for training (default {DEFAULT_N})")
    p.add_argument("--ood_per_source", type=int, default=DEFAULT_OOD,
                   help=f"OOD test reviews per source (default {DEFAULT_OOD})")
    p.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    p.add_argument("--output_dir",     type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_corpus(
        n_per_source   = args.n_per_source,
        ood_per_source = args.ood_per_source,
        seed           = args.seed,
        output_dir     = Path(args.output_dir),
    )
