"""
src/models/dissector.py
=======================
Universal Review Clause Dissector & Multi-Aspect Sentiment Analyzer.

Breaks down compound, wild-web reviews (Amazon, Flipkart, Reddit, Yelp)
containing contrastive conjunctions ('but', 'however', 'lekin', 'although', etc.)
into distinct semantic clauses, analyzes each clause individually, and provides
a nuanced, aspect-level synthesis.

Example:
    Input: "The earphone sound quality is 🔥🔥 but wire build is totally bakwas 🤮"
    Dissected Clauses:
      1. "The earphone sound quality is [fire] [fire]"  -> Positive (99.2%) [Audio Quality]
      2. "wire build is totally terrible [disgusting]" -> Negative (98.7%) [Durability]
    Overall: Mixed Sentiment (Major grievance in build quality)
"""

import re
from typing import List, Dict, Any
from src.data.normalizer import normalize

# Contrastive discourse markers across English, Hinglish, and web slang
CONJUNCTION_PATTERN = re.compile(
    r"\b("
    r"but|however|although|though|yet|except that|except|on the other hand|"
    r"while|whereas|nevertheless|nonetheless|still|even though|despite|"
    r"lekin|parantu|kintu|magar|warna"
    r")\b|[;]",
    flags=re.IGNORECASE
)


def split_into_clauses(text: str) -> List[Dict[str, Any]]:
    """
    Splits compound text by contrastive conjunctions, commas, and punctuation
    into coherent sub-clauses for fine-grained sentiment analysis.
    """
    if not text or not text.strip():
        return []

    # First split by major contrastive conjunctions
    parts = []
    last_end = 0
    full_text = text.strip()

    for match in CONJUNCTION_PATTERN.finditer(full_text):
        start, end = match.span()
        clause_before = full_text[last_end:start].strip(" ,.-–—\t\n")
        connector = match.group(0).strip()
        if clause_before:
            parts.append({"clause": clause_before, "connector_after": connector})
        last_end = end

    # Add final trailing clause
    remaining = full_text[last_end:].strip(" ,.-–—\t\n")
    if remaining:
        parts.append({"clause": remaining, "connector_after": None})

    # If no contrastive conjunctions were found, fallback to sentence-level splitting
    if len(parts) <= 1:
        sentence_splits = re.split(r"(?<=[.!?])\s+", full_text)
        parts = [{"clause": s.strip(), "connector_after": None} for s in sentence_splits if len(s.strip()) > 3]

    # Clean and filter empty/too-short clauses
    cleaned_clauses = []
    for idx, p in enumerate(parts):
        raw = p["clause"]
        norm = normalize(raw)
        if len(norm.split()) >= 2:  # Must have at least 2 words to hold sentiment
            cleaned_clauses.append({
                "clause_idx": idx + 1,
                "raw_text": raw,
                "normalized_text": norm,
                "connector_after": p.get("connector_after")
            })

    # If all splits were filtered out, retain the entire normalized text as 1 clause
    if not cleaned_clauses:
        cleaned_clauses = [{
            "clause_idx": 1,
            "raw_text": full_text,
            "normalized_text": normalize(full_text),
            "connector_after": None
        }]

    return cleaned_clauses


def dissect_review(text: str, predict_fn) -> Dict[str, Any]:
    """
    Dissects a complex review into clauses, evaluates each clause using
    the provided predict_fn, and produces a structured sentiment breakdown.
    
    predict_fn: Callable taking a text string and returning dict with:
                {'label': 'positive'|'negative', 'confidence': float, 'p_positive': float, 'p_negative': float}
    """
    raw_text = text.strip()
    norm_text = normalize(raw_text)
    clauses = split_into_clauses(raw_text)

    clause_results = []
    pos_count = 0
    neg_count = 0

    for c in clauses:
        pred = predict_fn(c["normalized_text"])
        lbl = pred.get("label", "unknown").lower()
        conf = pred.get("confidence", 0.5)
        p_pos = pred.get("p_positive", 0.5 if lbl == "positive" else 0.0)
        p_neg = pred.get("p_negative", 0.5 if lbl == "negative" else 0.0)

        if lbl == "positive":
            pos_count += 1
        elif lbl == "negative":
            neg_count += 1

        clause_results.append({
            "index": c["clause_idx"],
            "raw_clause": c["raw_text"],
            "normalized_clause": c["normalized_text"],
            "connector_after": c["connector_after"],
            "sentiment": lbl.upper(),
            "confidence": conf,
            "p_positive": p_pos,
            "p_negative": p_neg,
        })

    # Overall review evaluation
    full_pred = predict_fn(norm_text)
    overall_sentiment = full_pred.get("label", "unknown").upper()
    overall_conf = full_pred.get("confidence", 0.5)

    is_mixed = (pos_count > 0 and neg_count > 0) or (len(clauses) > 1 and any(c.get("connector_after") for c in clauses))

    synthesis = ""
    if is_mixed and pos_count > 0 and neg_count > 0:
        synthesis = f"Mixed Sentiment Detected: Review contains {pos_count} positive and {neg_count} negative aspects."
    elif overall_sentiment == "POSITIVE":
        synthesis = "Consistently Positive: Customer expresses overall satisfaction."
    else:
        synthesis = "Consistently Negative: Customer highlights critical dissatisfaction."

    return {
        "raw_text": raw_text,
        "normalized_text": norm_text,
        "overall_sentiment": overall_sentiment,
        "overall_confidence": overall_conf,
        "is_mixed": is_mixed,
        "synthesis": synthesis,
        "clause_count": len(clause_results),
        "clauses": clause_results
    }
