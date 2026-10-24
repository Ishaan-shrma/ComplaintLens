"""Paths and constants. Everything here is read-only configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv


def _find_root() -> Path:
    override = os.environ.get("REVIEW_AI_ROOT")          # used by tests
    if override:
        return Path(override).resolve()
    for d in Path(__file__).resolve().parents:
        if (d / "data" / "processed").exists() or (d / "data" / "sample").exists():
            return d
    raise FileNotFoundError("Project root (a folder containing data/processed or data/sample) not found.")


ROOT = _find_root()
load_dotenv(ROOT / ".env")

# Dataset Paths & Runtime Data Mode
REVIEWS_CSV = ROOT / "data" / "processed" / "reviews_cleaned.csv"
SAMPLE_REVIEWS_CSV = ROOT / "data" / "sample" / "sample_reviews_5k.csv"
DATA_MODE = "full" if REVIEWS_CSV.exists() else ("sample" if SAMPLE_REVIEWS_CSV.exists() else "none")
ACTIVE_REVIEWS_CSV = REVIEWS_CSV if REVIEWS_CSV.exists() else SAMPLE_REVIEWS_CSV

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

# LLM Provider Configuration: "ollama" | "api" | "none" (default: "none")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()

# Dynamic Base URL defaults per provider
_default_base_url = "http://localhost:11434" if LLM_PROVIDER == "ollama" else "https://api.openai.com/v1"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL" if LLM_PROVIDER == "ollama" else "LLM_API_BASE_URL", _default_base_url)).strip()

# Dynamic Model defaults per provider
_default_model = "llama3.2:3b" if LLM_PROVIDER == "ollama" else "gpt-4o-mini"
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL" if LLM_PROVIDER == "ollama" else "LLM_API_MODEL", _default_model)).strip()

# API Key strictly read on server - never exposed to browser or client
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()

# Agent Operational Limits & Rate Controls
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
LLM_DAILY_CALL_CAP = int(os.getenv("LLM_DAILY_CALL_CAP", "200"))
