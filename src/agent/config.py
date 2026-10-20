"""Paths and constants. Everything here is read-only configuration."""
import os
from pathlib import Path


def _find_root() -> Path:
    override = os.environ.get("REVIEW_AI_ROOT")          # used by tests
    if override:
        return Path(override).resolve()
    for d in Path(__file__).resolve().parents:
        if (d / "data" / "processed").exists():
            return d
    raise FileNotFoundError("Project root (a folder containing data/processed) not found.")


ROOT = _find_root()
REVIEWS_CSV = ROOT / "data" / "processed" / "reviews_cleaned.csv"
SAMPLE_META_CSV = ROOT / "data" / "processed" / "embedding_sample_metadata.csv"
UNIVERSAL_META_CSV = ROOT / "data" / "processed" / "universal_sample_metadata.csv"
FAISS_INDEX = ROOT / "models" / "faiss_index_with_ids.bin"
FAISS_UNIVERSAL_INDEX = ROOT / "models" / "faiss_universal_160k.bin"
ART = ROOT / "notebooks" / "artifacts"                   # change here if artifacts move to models/
SENTIMENT_DIR = ART / "distilbert_full281k"
DEBERTA_DIR = ROOT / "models" / "deberta_omni_v1"
TEMPERATURE_FILE = ART / "temperature_T.npy"

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_TOKENS = 128
MAX_INPUT_CHARS = 5000
MAX_TOP_K = 20

