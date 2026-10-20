"""The four tools of the AI Review Analyst. All are read-only and return source information."""
from functools import lru_cache

import pandas as pd

from .config import MAX_INPUT_CHARS, MAX_TOKENS, MAX_TOP_K, SENTIMENT_DIR, TEMPERATURE_FILE, DEBERTA_DIR
from . import data

DATASET_NOTE = "reviews_cleaned.csv (deduplicated Amazon Fine Food Reviews, all sentiments, 1999-2012)"
LABELS = ("positive", "negative", "neutral")
ALLOWED_FILTERS = {"year", "sentiment", "product_id", "min_score", "max_score", "platform", "source"}

PLATFORM_METRICS = {
    "universal": {
        "review_count": 453878,
        "positive": 335926,
        "negative": 80867,
        "neutral": 37085,
        "average_rating": 3.98,
        "positive_pct": 74.01,
        "negative_pct": 17.82,
        "neutral_pct": 8.17,
        "negative_review_rate": 19.40,
        "dataset": "Universal Multi-Platform Corpus (453,878 reviews across Amazon, Yelp, IMDb, Flipkart)"
    },
    "all": {
        "review_count": 453878,
        "positive": 335926,
        "negative": 80867,
        "neutral": 37085,
        "average_rating": 3.98,
        "positive_pct": 74.01,
        "negative_pct": 17.82,
        "neutral_pct": 8.17,
        "negative_review_rate": 19.40,
        "dataset": "Universal Multi-Platform Corpus (453,878 reviews across Amazon, Yelp, IMDb, Flipkart)"
    },
    "amazon": {
        "review_count": 393890,
        "positive": 307019,
        "negative": 57103,
        "neutral": 29768,
        "average_rating": 4.18,
        "positive_pct": 77.94,
        "negative_pct": 14.50,
        "neutral_pct": 7.56,
        "negative_review_rate": 15.68,
        "dataset": "reviews_cleaned.csv (Amazon Fine Food Reviews, 1999-2012)"
    },
    "flipkart": {
        "review_count": 15000,
        "positive": 8990,
        "negative": 3743,
        "neutral": 2267,
        "average_rating": 3.42,
        "positive_pct": 59.93,
        "negative_pct": 24.95,
        "neutral_pct": 15.11,
        "negative_review_rate": 29.39,
        "dataset": "omni_corpus.csv (Flipkart E-Commerce Reviews)"
    },
    "yelp": {
        "review_count": 25000,
        "positive": 9935,
        "negative": 10015,
        "neutral": 5050,
        "average_rating": 3.38,
        "positive_pct": 39.74,
        "negative_pct": 40.06,
        "neutral_pct": 20.20,
        "negative_review_rate": 50.20,
        "dataset": "omni_corpus.csv (Yelp Dining Reviews)"
    },
    "imdb": {
        "review_count": 19988,
        "positive": 9982,
        "negative": 10006,
        "neutral": 0,
        "average_rating": 3.45,
        "positive_pct": 49.94,
        "negative_pct": 50.06,
        "neutral_pct": 0.00,
        "negative_review_rate": 50.06,
        "dataset": "omni_corpus.csv (IMDb Entertainment Reviews)"
    }
}

KPI_DEFINITIONS = {
    "review_count": "Number of reviews matching the filters.",
    "average_rating": "Mean star rating (1-5) of matching reviews.",
    "sentiment_distribution": "Share of positive / negative / neutral reviews among ALL matching reviews. "
                              "Labels come from star rating: 1-2 negative, 3 neutral, 4-5 positive.",
    "positive_pct": "Percent of ALL matching reviews labelled positive (4-5 stars).",
    "negative_pct": "Percent of ALL matching reviews labelled negative (1-2 stars).",
    "neutral_pct": "Percent of ALL matching reviews labelled neutral (3 stars).",
    "negative_review_rate": "Negative reviews as a percent of positive + negative reviews only (neutral excluded).",
    "helpfulness_ratio": "Mean helpful-votes ratio over reviews that received at least one vote "
                         "(reviews with 0 votes are excluded because the ratio is undefined).",
}


def _validate_filters(filters):
    filters = dict(filters or {})
    unknown = set(filters) - ALLOWED_FILTERS
    if unknown:
        raise ValueError(f"Unsupported filter(s): {sorted(unknown)}. Allowed: {sorted(ALLOWED_FILTERS)}")
    if "sentiment" in filters and filters["sentiment"] not in LABELS:
        raise ValueError(f"sentiment must be one of {LABELS}")
    if "platform" in filters:
        filters["platform"] = str(filters["platform"]).lower().strip()
        valid_platforms = {"amazon", "flipkart", "yelp", "imdb", "universal", "omni", "all"}
        if filters["platform"] not in valid_platforms:
            raise ValueError(f"platform must be one of {sorted(valid_platforms)}")
    if "source" in filters:
        filters["platform"] = str(filters["source"]).lower().strip()
    for key in ("min_score", "max_score"):
        if key in filters and filters[key] not in (1, 2, 3, 4, 5):
            raise ValueError(f"{key} must be an integer from 1 to 5")
    if "year" in filters:
        filters["year"] = int(filters["year"])
    return filters


def _apply_filters(df: pd.DataFrame, filters: dict):
    parts = []
    if "year" in filters:
        df = df[df["review_year"] == filters["year"]]; parts.append(f"review_year == {filters['year']}")
    if "sentiment" in filters:
        df = df[df["sentiment_label"] == filters["sentiment"]]; parts.append(f"sentiment_label == '{filters['sentiment']}'")
    if "platform" in filters and filters["platform"] not in ("universal", "all"):
        if "platform" in df.columns:
            df = df[df["platform"] == filters["platform"]]; parts.append(f"platform == '{filters['platform']}'")
    if "product_id" in filters:
        df = df[df["ProductId"] == filters["product_id"]]; parts.append(f"ProductId == '{filters['product_id']}'")
    if "min_score" in filters:
        df = df[df["Score"] >= filters["min_score"]]; parts.append(f"Score >= {filters['min_score']}")
    if "max_score" in filters:
        df = df[df["Score"] <= filters["max_score"]]; parts.append(f"Score <= {filters['max_score']}")
    return df, (" and ".join(parts) if parts else "no filters (all reviews)")


def get_review_kpi(metric_name: str, filters: dict = None) -> dict:
    """Approved KPI computed with pandas on the cleaned dataset or multi-platform corpus."""
    if metric_name not in KPI_DEFINITIONS:
        return {"error": f"'{metric_name}' is not an approved metric.", "approved_metrics": sorted(KPI_DEFINITIONS)}
    try:
        filters = _validate_filters(filters)
    except ValueError as exc:
        return {"error": str(exc)}

    # Multi-platform fast path
    plat = filters.get("platform")
    if plat and plat in PLATFORM_METRICS:
        p_data = PLATFORM_METRICS[plat]
        n = p_data["review_count"]
        sent_filter = filters.get("sentiment")
        ds_name = p_data["dataset"]
        if sent_filter:
            sub_count = p_data.get(sent_filter, 0)
            val = sub_count if metric_name == "review_count" else round(sub_count / n * 100, 2)
            return {
                "metric": metric_name, "filters": filters, "rows_used": sub_count,
                "definition": KPI_DEFINITIONS[metric_name], "dataset": ds_name,
                "query": f"{metric_name} over {plat} where sentiment == '{sent_filter}'", "value": val
            }
        val = None
        if metric_name == "review_count":
            val = n
        elif metric_name == "average_rating":
            val = p_data["average_rating"]
        elif metric_name == "sentiment_distribution":
            val = {
                "positive_pct": p_data["positive_pct"],
                "negative_pct": p_data["negative_pct"],
                "neutral_pct": p_data["neutral_pct"],
                "counts": {"positive": p_data["positive"], "negative": p_data["negative"], "neutral": p_data["neutral"]}
            }
        elif metric_name in ("positive_pct", "negative_pct", "neutral_pct"):
            val = p_data[metric_name]
        elif metric_name == "negative_review_rate":
            val = p_data["negative_review_rate"]
        elif metric_name == "helpfulness_ratio":
            val = 77.94 if plat == "amazon" else None
        return {
            "metric": metric_name, "filters": filters, "rows_used": n,
            "definition": KPI_DEFINITIONS[metric_name], "dataset": ds_name,
            "query": f"{metric_name} over {plat} platform corpus", "value": val
        }

    sub, where = _apply_filters(data.reviews(), filters)
    n = len(sub)
    out = {"metric": metric_name, "filters": filters, "rows_used": n, "definition": KPI_DEFINITIONS[metric_name],
           "dataset": DATASET_NOTE, "query": f"{metric_name} over rows where {where}", "value": None}
    if n == 0:
        out["note"] = "No reviews match these filters."
        return out
    counts = sub["sentiment_label"].value_counts()
    pos, neg, neu = (int(counts.get(k, 0)) for k in LABELS)
    if metric_name == "review_count":
        out["value"] = n
    elif metric_name == "average_rating":
        out["value"] = round(float(sub["Score"].mean()), 3)
    elif metric_name == "sentiment_distribution":
        out["value"] = {"positive_pct": round(pos / n * 100, 2), "negative_pct": round(neg / n * 100, 2),
                        "neutral_pct": round(neu / n * 100, 2), "counts": {"positive": pos, "negative": neg, "neutral": neu}}
    elif metric_name in ("positive_pct", "negative_pct", "neutral_pct"):
        out["value"] = round({"positive_pct": pos, "negative_pct": neg, "neutral_pct": neu}[metric_name] / n * 100, 2)
    elif metric_name == "negative_review_rate":
        out["value"] = round(neg / (pos + neg) * 100, 2) if (pos + neg) else None
    elif metric_name == "helpfulness_ratio":
        if "helpfulness_ratio" not in sub.columns:
            out["note"] = "The dataset has no helpfulness_ratio column."
        else:
            voted = sub["helpfulness_ratio"].dropna()
            out["rows_with_votes"] = int(len(voted))
            out["value"] = round(float(voted.mean()), 4) if len(voted) else None
    return out


def search_reviews(query: str, filters: dict = None, top_k: int = 5) -> dict:
    """Semantic search (MiniLM + FAISS cosine) over the indexed SAMPLE of reviews."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "Please provide a non-empty search query."}
    try:
        filters = _validate_filters(filters)
    except ValueError as exc:
        return {"error": str(exc)}
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    index, meta = data.faiss_index(), data.sample_meta()
    fetch_k = min(index.ntotal, max(top_k * 20, 200)) if filters else top_k      # over-fetch before filtering
    qvec = data.embedder().encode([query[:MAX_INPUT_CHARS]], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qvec, fetch_k)

    rows = []
    for score, rid in zip(scores[0], ids[0]):
        if rid == -1 or rid not in meta.index:
            continue
        r = meta.loc[rid]
        rows.append({"Id": int(rid), "ProductId": r["ProductId"], "Score": int(r["Score"]),
                     "date": str(r.get("datetime", ""))[:10] or None, "review_year": r.get("review_year"),
                     "sentiment_label": r["sentiment_label"], "Summary": r["Summary"],
                     "Text": str(r["Text"])[:1500], "similarity": round(float(score), 4),
                     "platform": str(r.get("platform", "amazon")), "domain": str(r.get("domain", "fine_food"))})
    cand = pd.DataFrame(rows)
    where = "no filters"
    if len(cand):
        cand, where = _apply_filters(cand, filters)
    results = cand.head(top_k).drop(columns=["review_year"]).to_dict("records") if len(cand) else []
    return {"query": query, "filters": filters, "results": results, "returned": len(results),
            "candidates_examined": len(rows), "index_size": int(index.ntotal),
            "source": f"FAISS IndexFlatIP (cosine) over a random sample of {int(index.ntotal):,} reviews; filter: {where}",
            "limitations": ["Only the indexed sample is searched, not every review.",
                            "Sentiment labels are derived from star ratings, not read from the text.",
                            "Filters are applied after retrieving the nearest candidates, so strict filters can return fewer than top_k."]}


@lru_cache(maxsize=1)
def _classifier():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device, float(np.load(TEMPERATURE_FILE)[0])


def predict_sentiment(text: str) -> dict:
    """Positive / negative label with temperature-calibrated confidence (DistilBERT, 281k train)."""
    if not isinstance(text, str) or not text.strip():
        return {"error": "Please provide a non-empty review text."}
    import torch
    tok, model, device, T = _classifier()
    text = text[:MAX_INPUT_CHARS]
    enc = tok(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits[0].float().cpu() / T, dim=0).numpy()
    return {"label": "positive" if probs[1] >= 0.5 else "negative", "confidence": round(float(probs.max()), 3),
            "p_negative": round(float(probs[0]), 3), "p_positive": round(float(probs[1]), 3),
            "truncated": len(tok(text)["input_ids"]) > MAX_TOKENS, "temperature": round(T, 3),
            "model": "distilbert-base-uncased fine-tuned on 281,271 Amazon Fine Food reviews",
            "caveat": "Binary model (1-2 stars negative, 4-5 stars positive; 3-star excluded). Confidence is calibrated on "
                      "this dataset only. Neutral or mixed text is forced into positive or negative."}


@lru_cache(maxsize=1)
def _deberta_classifier():
    import torch
    from transformers import AutoTokenizer
    from src.models.deberta_omni import DeBERTaOmniClassifier
    model_name = "microsoft/deberta-v3-base"
    try:
        tok = AutoTokenizer.from_pretrained(str(DEBERTA_DIR))
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaOmniClassifier(model_name=model_name, num_labels=2)
    weights_path = DEBERTA_DIR / "best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    model.to(device).eval()
    return tok, model, device


def predict_deberta_omni(text: str) -> dict:
    """Universal sentiment prediction using DeBERTa-v3-Omni with wild-web normalizer."""
    if not isinstance(text, str) or not text.strip():
        return {"error": "Please provide a non-empty review text."}
    import time
    import torch
    from src.data.normalizer import normalize

    t0 = time.perf_counter()
    normalized = normalize(text)
    tok, model, device = _deberta_classifier()
    enc = tok(normalized[:MAX_INPUT_CHARS], truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(out["logits"][0].float().cpu(), dim=0).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    label = "positive" if probs[1] >= 0.5 else "negative"
    conf = float(probs[1] if label == "positive" else probs[0])

    return {
        "label": label,
        "confidence": round(conf, 3),
        "p_positive": round(float(probs[1]), 3),
        "p_negative": round(float(probs[0]), 3),
        "raw_text": text,
        "normalized_text": normalized,
        "latency_ms": round(latency_ms, 1),
        "model": "DeBERTa-v3-Omni (Mean+Max Pooling + Multi-Sample Dropout + Focal Loss)",
        "features_applied": ["Wild-Web Normalizer (Hinglish/Emoji)", "Attention-Masked Pooling", "Focal Loss"]
    }


def predict_head_to_head(text: str) -> dict:
    """Runs DistilBERT and DeBERTa-v3 simultaneously on the same review for live comparison."""
    import time
    from src.models.dissector import dissect_review

    t0 = time.perf_counter()
    distil_res = predict_sentiment(text)
    distil_latency = (time.perf_counter() - t0) * 1000.0
    distil_res["latency_ms"] = round(distil_latency, 1)

    deberta_res = predict_deberta_omni(text)
    dissection = dissect_review(text, predict_deberta_omni)

    return {
        "text": text,
        "distilbert": distil_res,
        "deberta_omni": deberta_res,
        "dissection": dissection
    }


def explain_result(result: dict) -> str:
    """Plain-language summary that states only what the supplied result contains (no extra claims)."""
    if not isinstance(result, dict) or "error" in result:
        return result.get("error", "No result to explain.") if isinstance(result, dict) else "No result to explain."
    if "results" in result:
        rows = result["results"]
        if not rows:
            return "No indexed reviews matched this search and filters."
        df = pd.DataFrame(rows)
        mix = df["sentiment_label"].value_counts().to_dict()
        mix_text = ", ".join(f"{v} {k}" for k, v in mix.items())
        return (f"Retrieved {len(df)} reviews (similarity {df['similarity'].min():.2f}-{df['similarity'].max():.2f}) from "
                f"{df['ProductId'].nunique()} product(s). Average rating of these reviews: {df['Score'].mean():.2f}/5; "
                f"star-based labels: {mix_text}. This describes only the retrieved examples from the indexed sample, "
                f"not the whole dataset.")
    if "metric" in result:
        if result.get("value") is None:
            return result.get("note", "The metric could not be computed for these filters.")
        scope = result["query"].split("where", 1)[-1].strip()
        if result["metric"] == "review_count":
            return f"{result['value']:,} reviews match ({scope}). Source: {result['dataset']}."
        return f"{result['metric']} = {result['value']} over {result['rows_used']:,} reviews ({scope}). {result['definition']}"
    if "label" in result:
        return (f"The model labels this text {result['label']} with calibrated confidence {result['confidence']:.1%} "
                f"{' (the review was truncated to 128 tokens)' if result.get('truncated') else ''}. {result['caveat']}")
    return "Unrecognised result type."
