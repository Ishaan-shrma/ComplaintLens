"""Request guardrails: block anything that is not a read-only question about the review data."""
import re

MAX_QUESTION_CHARS = 2000

# (category, regex, message). First match wins; checks run on the lower-cased question.
RULES = [
    ("prompt_injection",
     r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules)"
     r"|disregard (the |all |your )?(rules|instructions|guardrails)|system prompt|developer mode|jailbreak",
     "I can't change my rules or instructions."),
    ("sql",
     r"\bselect\b.+\bfrom\b|\binsert\s+into\b|\bupdate\b.+\bset\b|\bdelete\s+from\b"
     r"|\bdrop\s+(table|database)\b|\btruncate\s+table\b|\balter\s+table\b|\bcreate\s+table\b",
     "I don't run SQL. Ask a question in plain language instead."),
    ("write",
     r"\b(delete|erase|wipe|truncate|overwrite)\b"
     r"|\b(edit|modify|change|update|remove|add|insert|rename|replace)\b.{0,40}"
     r"\b(review|reviews|rating|ratings|label|labels|score|scores|dataset|database|table|row|rows|record|records|column|data)\b",
     "This assistant is read-only. It cannot add, edit or delete reviews, labels or ratings."),
    ("secrets",
     r"api[ _-]?key|password|passwd|secret|credential|access[ _-]?token|private key|\.env\b|\bssh\b",
     "I can't access or reveal secrets, keys or credentials."),
    ("system_access",
     r"os\.system|subprocess|\beval\(|\bexec\(|rm\s+-rf|\.\./|\b[a-z]:\\|/etc/|/home/|powershell"
     r"|read (the )?file|open (the )?file|list (the )?(files|folders|directory)",
     "I can't access files, folders or run commands. I only answer from the review dataset."),
    ("bulk_export",
     r"\b(export|dump|download|exfiltrate)\b|\b(entire|whole|full|complete) (dataset|database|table|csv)"
     r"|all (the )?reviews (as|in) (csv|json)",
     "Bulk export of raw data is not allowed. I can return a few cited example reviews or aggregate KPIs."),
    ("personal_data",
     r"\buser ?id\b|\bprofile ?name\b|\bwho (wrote|posted|reviewed)\b|which (user|customer|reviewer)"
     r"|\bemail address\b|\breviewer'?s? (name|identity)",
     "Reviewer identities are not exposed. I can answer questions about the reviews themselves."),
    ("out_of_scope",
     r"\bforecast|\bwill (sales|ratings|reviews)\b|next (year|month|quarter)|\bsales\b|\brevenue\b"
     r"|\bprofit\b|stock price|market share|competitor",
     "That is outside the data: the dataset has reviews only (1999-2012), no sales, revenue or forecasts."),
]
_COMPILED = [(c, re.compile(p), m) for c, p, m in RULES]


def check_request(text) -> dict:
    """Return {'allowed': bool, 'category': str|None, 'message': str|None}."""
    if not isinstance(text, str) or not text.strip():
        return {"allowed": False, "category": "empty", "message": "Please type a question."}
    if len(text) > MAX_QUESTION_CHARS:
        return {"allowed": False, "category": "too_long", "message": f"Please keep questions under {MAX_QUESTION_CHARS} characters."}
    low = text.lower()
    for category, pattern, message in _COMPILED:
        if pattern.search(low):
            return {"allowed": False, "category": category, "message": message}
    return {"allowed": True, "category": None, "message": None}
