"""
src/data/normalizer.py
=======================
Wild-Web Text Normalizer — cleans reviews from any website on the internet
before they hit the model.

Handles:
  - Emoji → semantic text tokens   (🔥 → "[fire]", 💀 → "[dead]")
  - Hinglish & modern slang        ("bakwas" → "terrible", "goated" → "greatest")
  - Explicit rating strings        ("2/10", "1 out of 5") → anchored sentiment tokens
  - Character elongation           ("sooooo goooood" → "so good")
  - HTML / URL stripping
  - Whitespace normalisation

Usage
-----
    from src.data.normalizer import normalize
    clean = normalize("This product is 🔥🔥 but delivery was bakwas 💀")
    # → "This product is [fire] [fire] but delivery was terrible [dead]"
"""

import re
import unicodedata
from typing import Optional

# ── Emoji → semantic token map ───────────────────────────────────────────────
# Covers the most frequent review emojis; extend freely.
EMOJI_MAP: dict[str, str] = {
    # Positive
    "🔥": "[fire]",
    "💯": "[perfect]",
    "❤️": "[love]",
    "❤":  "[love]",
    "😍": "[love]",
    "🥰": "[love]",
    "😊": "[happy]",
    "😁": "[happy]",
    "😄": "[happy]",
    "👍": "[thumbs_up]",
    "👌": "[ok]",
    "✅": "[check]",
    "🙌": "[praise]",
    "🏆": "[trophy]",
    "⭐": "[star]",
    "🌟": "[star]",
    "✨": "[sparkle]",
    "💪": "[strong]",
    "🎉": "[celebration]",
    "😎": "[cool]",
    "🤩": "[amazing]",
    "💚": "[love]",
    "💙": "[love]",
    "🫶": "[love]",
    "🙏": "[thanks]",
    "W":  "",            # "W purchase" handled by slang map

    # Negative
    "💀": "[dead]",
    "🗑️": "[trash]",
    "🗑":  "[trash]",
    "🤮": "[disgusting]",
    "🤢": "[disgusting]",
    "😡": "[angry]",
    "😠": "[angry]",
    "😤": "[frustrated]",
    "👎": "[thumbs_down]",
    "❌": "[wrong]",
    "🚫": "[banned]",
    "😒": "[disappointed]",
    "😞": "[disappointed]",
    "😢": "[sad]",
    "😭": "[crying]",
    "🥲": "[sad]",
    "💸": "[wasted_money]",
    "⚠️": "[warning]",
    "⚠":  "[warning]",
    "🙄": "[eyeroll]",

    # Neutral / mixed
    "🤔": "[thinking]",
    "😐": "[neutral]",
    "🫤": "[neutral]",
    "📦": "[package]",
    "🚚": "[delivery]",
    "💰": "[money]",
    "📱": "[phone]",
    "💻": "[laptop]",
    "🍕": "[food]",
    "☕": "[coffee]",
    "🍔": "[food]",
}

# ── Slang / Hinglish → English expansions ────────────────────────────────────
# Lower-cased whole-word replacements (applied after lowercasing).
SLANG_MAP: dict[str, str] = {
    # Hinglish — negative
    "bakwas":        "terrible",
    "bekar":         "useless",
    "khatam":        "finished",
    "faltu":         "worthless",
    "ghatiya":       "inferior",
    "nakli":         "fake",
    "dhoka":         "fraud",
    "cheat":         "fraud",       # common in Indian e-comm reviews
    "total waste":   "complete waste of money",

    # Hinglish — positive
    "paisa vasool":  "worth every penny",
    "ekdum mast":    "absolutely great",
    "mast":          "great",
    "zabardast":     "excellent",
    "jhakaas":       "fantastic",
    "bindaas":       "carefree and good",
    "sahi hai":      "it is good",
    "badhiya":       "very good",
    "kadak":         "strong and good",

    # Modern internet slang — positive
    "goated":        "greatest of all time",
    "goat":          "greatest",
    "w purchase":    "great purchase",
    "w product":     "great product",
    "based":         "admirable",
    "bussin":        "extremely good",
    "bussin bussin": "extremely good",
    "fire":          "excellent",    # "this is fire" (text form)
    "slaps":         "sounds or tastes great",
    "hits different":"feels special",
    "no cap":        "honestly",
    "fr fr":         "seriously",
    "lowkey":        "somewhat",
    "highkey":       "very much",
    "slay":          "excellent job",
    "ate":           "did perfectly",
    "understood the assignment": "did perfectly",
    "rent free":     "unforgettable",

    # Modern internet slang — negative
    "mid":           "mediocre",
    "L product":     "bad product",
    "l product":     "bad product",
    "ngl":           "honestly",
    "dog water":     "very bad",
    "trash":         "very bad",
    "dogshit":       "terrible",
    "sus":           "suspicious",
    "cap":           "lie",
    "ratio":         "unpopular opinion",
    "rip":           "ruined",
    "bricked":       "broken",
    "dead product":  "broken product",

    # Abbreviations common in reviews
    "tbh":   "to be honest",
    "imo":   "in my opinion",
    "imho":  "in my humble opinion",
    "omg":   "oh my god",
    "smh":   "shaking my head",
    "istg":  "i swear to god",
    "idk":   "i do not know",
    "btw":   "by the way",
    "asap":  "as soon as possible",
    "fwiw":  "for what it is worth",
    "lmk":   "let me know",
    "ty":    "thank you",
    "thx":   "thanks",
    "pls":   "please",
    "plz":   "please",
    "ur":    "your",
    "u":     "you",
    "r":     "are",
    "n":     "and",
    "gg":    "good",
    "af":    "very",       # "bad af" → "bad very" (imperfect but useful)
    "rn":    "right now",
}

# ── Explicit rating patterns → sentiment anchors ──────────────────────────────
# Matched before lowercasing so "1/5" etc. are caught.
# Each tuple: (compiled regex, replacement_string)
_RATING_PATTERNS = [
    # "X/10" where X ∈ 0-10
    (re.compile(r"\b([0-9]|10)\s*/\s*10\b"), lambda m: _score_token(int(m.group(1)), 10)),
    # "X/5"  where X ∈ 0-5
    (re.compile(r"\b([0-5])\s*/\s*5\b"),     lambda m: _score_token(int(m.group(1)), 5)),
    # "X out of 10" / "X out of 5"
    (re.compile(r"\b([0-9]|10)\s+out\s+of\s+10\b", re.IGNORECASE),
     lambda m: _score_token(int(m.group(1)), 10)),
    (re.compile(r"\b([0-5])\s+out\s+of\s+5\b", re.IGNORECASE),
     lambda m: _score_token(int(m.group(1)), 5)),
    # "X stars"
    (re.compile(r"\b([1-5])\s*(?:star|stars)\b", re.IGNORECASE),
     lambda m: _score_token(int(m.group(1)), 5)),
]


def _score_token(score: int, max_score: int) -> str:
    """Convert a numeric score to a qualitative sentiment anchor token."""
    pct = score / max_score
    if pct >= 0.8:
        return "[excellent_rating]"
    elif pct >= 0.6:
        return "[good_rating]"
    elif pct >= 0.4:
        return "[average_rating]"
    elif pct >= 0.2:
        return "[poor_rating]"
    else:
        return "[terrible_rating]"


# ── Elongation compressor ─────────────────────────────────────────────────────
# "sooooo" → "so",  "greaaaat" → "great"  (keeps at most 2 repeated chars)
_ELONGATION_RE = re.compile(r"(.)\1{2,}")


def _compress_elongation(text: str) -> str:
    return _ELONGATION_RE.sub(r"\1\1", text)


# ── HTML & URL stripper ────────────────────────────────────────────────────────
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE  = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


# ── Punctuation normaliser ────────────────────────────────────────────────────
# Collapses runs of punctuation ("!!!" → "!", "..." → "…")
_MULTI_EXCLAIM = re.compile(r"!{2,}")
_MULTI_QUESTION = re.compile(r"\?{2,}")
_MULTI_DOT = re.compile(r"\.{3,}")


# ── Public API ────────────────────────────────────────────────────────────────

def normalize(
    text: str,
    *,
    expand_slang: bool = True,
    convert_emojis: bool = True,
    convert_ratings: bool = True,
    compress_elongation: bool = True,
    strip_html: bool = True,
    lowercase_output: bool = False,
) -> str:
    """
    Full normalization pipeline.

    Parameters
    ----------
    text                : Raw review string.
    expand_slang        : Replace Hinglish / internet slang with English.
    convert_emojis      : Replace emojis with semantic tokens.
    convert_ratings     : Convert "2/10", "1 star" etc. to tokens.
    compress_elongation : Collapse "sooooo" → "soo".
    strip_html          : Remove HTML tags and URLs.
    lowercase_output    : Return lowercase string (default False; tokenizer handles casing).

    Returns
    -------
    Cleaned string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 0. Strip HTML tags and URLs
    if strip_html:
        text = _HTML_RE.sub(" ", text)
        text = _URL_RE.sub(" [url] ", text)

    # 1. Rating patterns (before lowercasing so "1/5" is safe)
    if convert_ratings:
        for pattern, repl in _RATING_PATTERNS:
            text = pattern.sub(repl, text)

    # 2. Emoji substitution (unicode-aware, before lowercasing)
    if convert_emojis:
        for emoji, token in EMOJI_MAP.items():
            text = text.replace(emoji, f" {token} " if token else " ")

    # 3. Normalise unicode (e.g. fancy apostrophes, en-dashes)
    text = unicodedata.normalize("NFKC", text)

    # 4. Elongation compression
    if compress_elongation:
        text = _compress_elongation(text)

    # 5. Punctuation normalisation
    text = _MULTI_EXCLAIM.sub("!", text)
    text = _MULTI_QUESTION.sub("?", text)
    text = _MULTI_DOT.sub("...", text)

    # 6. Slang / Hinglish expansion (whole-word, case-insensitive)
    if expand_slang:
        # Work on lowercased copy for matching, then reconstruct
        lower = text.lower()
        for slang, expansion in SLANG_MAP.items():
            # Use word-boundary regex for accurate whole-word match
            pattern = r"(?<!\w)" + re.escape(slang) + r"(?!\w)"
            lower = re.sub(pattern, expansion, lower)
        text = lower  # output is lowercase after slang expansion

    # 7. Whitespace normalisation
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Optional explicit lowercase
    if lowercase_output and not expand_slang:
        text = text.lower()

    return text


def batch_normalize(texts: list[str], **kwargs) -> list[str]:
    """Vectorised wrapper — normalizes a list of review strings."""
    return [normalize(t, **kwargs) for t in texts]


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_CASES = [
        ("Emoji positive",     "This product is 🔥🔥 absolutely loved it 💯"),
        ("Emoji negative",     "Total 🗑️ arrived broken 💀 waste of money"),
        ("Hinglish negative",  "Ekdum bakwas product hai, paisa vasool nahi hua"),
        ("Hinglish positive",  "Zabardast quality! Paisa vasool product ekdum mast"),
        ("Modern slang pos",   "This keyboard is literally goated no cap, bussin fr fr"),
        ("Modern slang neg",   "Mid af product tbh, total L. Dog water quality smh"),
        ("Rating string",      "I give this 2/10. Bought 4 stars expecting quality."),
        ("Elongation",         "Sooooo goooood!!! Loved itttttt so much!!!!!!"),
        ("Mixed sarcasm",      "Oh wow greaaaat product, only took 3 weeks to arrive 🙄"),
        ("HTML+URL",           'Check <b>this</b> out at https://example.com great stuff'),
        ("Code-switched",      "Delivery was fast but product quality bekar hai, ghatiya packaging"),
    ]

    print("\n--- Wild-Web Normalizer Self-Test ---\n")
    for label, raw in TEST_CASES:
        cleaned = normalize(raw)
        print(f"[{label}]")
        print(f"  IN : {raw}")
        print(f"  OUT: {cleaned}\n")
