"""Router: guardrails -> choose a tool -> grounded answer with sources and limitations."""
import re

from . import tools
from .guardrails import check_request

LIMITATIONS = ["Read-only: nothing is written or changed.",
               "Answers come only from the Amazon Fine Food Reviews dataset (1999-2012); star-based sentiment labels are approximate.",
               "Semantic search covers an indexed random sample, not every review."]

_PREDICT = re.compile(r"predict(ing)?\s+(the\s+)?sentiment|classify\s+(this|the)\s+review|sentiment\s+of\s+(this|the)|is\s+this\s+(review\s+)?(positive|negative)")
_KPI = [(r"helpful", "helpfulness_ratio"),
        (r"average (star )?(rating|score)|avg (rating|score)|mean (rating|score)", "average_rating"),
        (r"negative (review )?rate|rate of negative", "negative_review_rate"),
        (r"(percent|percentage|share|proportion|%|fraction).{0,30}negative|negative.{0,30}(percent|percentage|share|proportion|%)", "negative_pct"),
        (r"(percent|percentage|share|proportion|%|fraction).{0,30}positive|positive.{0,30}(percent|percentage|share|proportion|%)", "positive_pct"),
        (r"(percent|percentage|share|proportion|%|fraction).{0,30}neutral|neutral.{0,30}(percent|percentage|share|proportion|%)", "neutral_pct"),
        (r"distribution|breakdown|sentiment (mix|split)", "sentiment_distribution"),
        (r"how many|number of|\bcount\b|total reviews|review volume", "review_count")]
_TEXT_TOPIC = re.compile(r"\b(mention|mentions|mentioning|about|containing|regarding|talk(ing)? about)\b")
_SEARCH_INTENT = re.compile(r"\b(customers|people|reviewers|buyers|users) (say|saying|think|complain|complaining|mention|mentioning)"
                            r"|\bsimilar\b|\bfind\b|\bshow me\b|\bcomplaints?\b|\bexamples?\b")
_SENTIMENT_METRICS = {"sentiment_distribution", "positive_pct", "negative_pct", "neutral_pct", "negative_review_rate"}


def _split_payload(question: str):
    m = re.search(r'["\u201c](.+?)["\u201d]', question, re.S)
    if m:
        return question[:m.start()], m.group(1)
    if ":" in question:
        head, tail = question.split(":", 1)
        return head, tail.strip()
    return question, ""


def parse_filters(question: str) -> dict:
    low, f = question.lower(), {}
    m = re.search(r"\b(19|20)\d{2}\b", low)
    if m and 1999 <= int(m.group()) <= 2012:
        f["year"] = int(m.group())
    for label in tools.LABELS:
        if re.search(rf"\b{label}\b", low):
            f["sentiment"] = label
    m = re.search(r"\b(B[0-9A-Z]{9})\b", question)
    if m:
        f["product_id"] = m.group(1)
    m = re.search(r"\b([1-5])[- ]?stars?\b", low)
    if m:
        f["min_score"] = f["max_score"] = int(m.group(1))
    return f


def answer(question: str) -> dict:
    """Answer one question. Always returns a dict with a 'status' of ok, blocked, unsupported, needs_input or error."""
    base = {"question": question, "limitations": LIMITATIONS}
    low = question.lower() if isinstance(question, str) else ""
    is_predict = bool(_PREDICT.search(low))
    instruction, payload = _split_payload(question) if is_predict else (question, "")
    guard = check_request(instruction)                      # pasted review text is data, not instructions
    if not guard["allowed"]:
        return {**base, "status": "blocked", "category": guard["category"], "explanation": guard["message"]}
    try:
        if is_predict:
            if not payload:
                return {**base, "status": "needs_input", "explanation": "Paste the review text after a colon or in quotes."}
            res = tools.predict_sentiment(payload)
            return {**base, "status": "ok", "tool": "predict_sentiment", "tool_input": {"text": payload[:200]},
                    "result": res, "explanation": tools.explain_result(res), "sources": [res.get("model")]}
        if re.search(r"\b(theme|themes|topic|topics|cluster|clusters)\b", low):
            return {**base, "status": "unsupported", "explanation": "Theme-level sentiment is not available: cluster assignments are not "
                    "stored per review. Try a semantic search such as 'complaints about packaging'."}
        if re.search(r"\b(trend|over time|has changed|have changed)\b", low) and "year" not in parse_filters(question):
            return {**base, "status": "unsupported", "explanation": "Ask for a specific year, for example 'average rating in 2010'."}
        metric = next((m for pattern, m in _KPI if re.search(pattern, low)), None)
        if metric and not _SEARCH_INTENT.search(low):
            if _TEXT_TOPIC.search(low):
                return {**base, "status": "unsupported", "explanation": "KPIs cover structured fields, not text mentions. "
                        "Use a semantic search to find reviews about a topic."}
            filters = parse_filters(question)
            if metric in _SENTIMENT_METRICS:
                filters.pop("sentiment", None)
            res = tools.get_review_kpi(metric, filters)
            return {**base, "status": "ok", "tool": "get_review_kpi", "tool_input": {"metric_name": metric, "filters": filters},
                    "result": res, "explanation": tools.explain_result(res), "sources": [res.get("query"), res.get("dataset")]}
        filters = parse_filters(question)
        res = tools.search_reviews(question, filters, top_k=5)
        return {**base, "status": "ok", "tool": "search_reviews", "tool_input": {"query": question, "filters": filters},
                "result": res, "explanation": tools.explain_result(res),
                "sources": [f"review Id {r['Id']}" for r in res.get("results", [])] + [res.get("source")]}
    except Exception as exc:                                # keep the app alive; no stack traces to the user
        return {**base, "status": "error", "explanation": f"The tool failed: {exc}"}
