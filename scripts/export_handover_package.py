"""
scripts/export_handover_package.py
==================================
Packages a clean, self-contained DS Group Internship Handover folder
suitable for a 16 GB USB pendrive (~3.8 GB total, compressing to ~2 GB ZIP).
Excludes the 72+ GB of disposable bloat:
  - .git/ (30.6 GB)
  - scratch/ (36.7 GB training checkpoints and optimizer states)
  - .venv/ (5.4 GB virtual environment)
  - node_modules/ (95 MB dependencies, pre-built dist/ is included)
"""

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
TARGET_DIR = Path("A:/DS-Review-AI-Handover")

# Essential files and directories to copy
ITEMS_TO_COPY = [
    # Source and server
    ("src", "src"),
    ("server", "server"),
    ("dist", "dist"),
    ("public", "public"),
    ("scripts", "scripts"),
    ("tests", "tests"),
    ("dashboards", "dashboards"),
    ("presentations", "presentations"),
    ("reports", "reports"),
    
    # Models (production weights & vector index only)
    ("models/deberta_omni_v1", "models/deberta_omni_v1"),
    ("models/faiss", "models/faiss"),
    ("models/faiss_universal_160k.bin", "models/faiss_universal_160k.bin"),
    ("models/svm_pipeline.joblib", "models/svm_pipeline.joblib"),
    ("models/sentiment_model_final.joblib", "models/sentiment_model_final.joblib"),
    ("models/sentiment_model_refined.joblib", "models/sentiment_model_refined.joblib"),

    # Notebooks & artifacts
    ("notebooks/01_data_understanding.ipynb", "notebooks/01_data_understanding.ipynb"),
    ("notebooks/01 data understanding week1.ipynb", "notebooks/01 data understanding week1.ipynb"),
    ("notebooks/02_nlp_eda.ipynb", "notebooks/02_nlp_eda.ipynb"),
    ("notebooks/03_sentiment_model_final.ipynb", "notebooks/03_sentiment_model_final.ipynb"),
    ("notebooks/03b_distilbert_finetuning.ipynb", "notebooks/03b_distilbert_finetuning.ipynb"),
    ("notebooks/04_embeddings_retrieval.ipynb", "notebooks/04_embeddings_retrieval.ipynb"),
    ("notebooks/05_phase2_deberta_omni_training.ipynb", "notebooks/05_phase2_deberta_omni_training.ipynb"),
    ("notebooks/artifacts/distilbert_full281k", "notebooks/artifacts/distilbert_full281k"),
    ("notebooks/artifacts/test_72846.pkl", "notebooks/artifacts/test_72846.pkl"),
    ("notebooks/artifacts/full_train_281271.pkl", "notebooks/artifacts/full_train_281271.pkl"),
    ("notebooks/artifacts/temperature_T.npy", "notebooks/artifacts/temperature_T.npy"),
    ("notebooks/artifacts/distilbert281k_test_logits.npy", "notebooks/artifacts/distilbert281k_test_logits.npy"),

    # Datasets
    ("data/raw/Reviews.csv", "data/raw/Reviews.csv"),
    ("data/processed/reviews_cleaned.csv", "data/processed/reviews_cleaned.csv"),
    ("data/processed/omni_corpus.csv", "data/processed/omni_corpus.csv"),
    ("data/processed/split_indices.npz", "data/processed/split_indices.npz"),

    # Documentation and root configs
    ("package.json", "package.json"),
    ("requirements.txt", "requirements.txt"),
    ("README.md", "README.md"),
    ("docs/report.md", "docs/report.md"),
    ("docs/DISCREPANCIES.md", "docs/DISCREPANCIES.md"),
    ("ML_Systems_Technical_Defense_Manual.pdf", "ML_Systems_Technical_Defense_Manual.pdf"),
    ("vite.config.js", "vite.config.js"),
    ("tailwind.config.js", "tailwind.config.js"),
]

BAT_LAUNCHER = '''@echo off
title DS Group Review Intelligence Prototype - Live Demo
echo ====================================================================
echo   DS GROUP REVIEW INTELLIGENCE PROTOTYPE (LIVE DEMO LAUNCHER)
echo   Model: DeBERTa-v3-Omni (#1 Champion) ^& DistilBERT 281k (#2 Runner-Up)
echo ====================================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ to run the backend.
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import fastapi, torch, transformers" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing required minimal packages...
    pip install fastapi uvicorn torch transformers pydantic scikit-learn faiss-cpu
)

echo [2/3] Starting Unified Backend ^& Standalone UI on http://127.0.0.1:8000 ...
echo [INFO] The browser will open automatically in 3 seconds.
start "" http://127.0.0.1:8000

python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
'''

README_HANDOVER = '''# DS Group Review Intelligence Prototype — Project Handover

**Author:** Ishaan Sharma (AI/ML Engineering Intern)  
**Organization:** Dharampal Satyapal Group (DS Group)  
**Project Track:** AI/ML Engineering, NLP Sentiment Modeling and Review Intelligence Agent  

---

### How to Run the Live Demo on Any Machine
1. Ensure Python 3.10+ is installed.
2. Double-click `run_demo.bat` (Windows) or execute:
   ```bash
   pip install -r requirements.txt
   python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
   ```
3. Open your browser at `http://127.0.0.1:8000` to interact with the full 7-workspace enterprise UI, live DeBERTa-v3-Omni predictions, calibrated Platt temperature scaling, semantic vector search, and read-only AI agent.

---

### Package Contents
- `src/`: Complete PyTorch model definitions, Disentangled Attention pooling head, agent tools, guardrails.
- `server/`: FastAPI read-only server.
- `dist/`: Pre-compiled React 18 production frontend (operates standalone with zero Node.js setup needed).
- `models/`: Trained DeBERTa-v3-Omni production checkpoint (`best_model.pt`), Linear SVM pipeline, FAISS index.
- `notebooks/`: All 7 documented Jupyter experiment notebooks and DistilBERT model weights.
- `data/`: Cleaned review datasets (`reviews_cleaned.csv`, `omni_corpus.csv`) and raw Kaggle input.
- `reports/`: Complete statistical verification reports, per-row predictions CSV, and multi-domain benchmarks.
- `docs/report.md`: Full 45-day internship work log and results summary.
- `docs/DISCREPANCIES.md`: Discrepancy log adhering to Rule R3.
- `ML_Systems_Technical_Defense_Manual.pdf`: Technical architecture and model defense guide.
'''

def build_package():
    print(f"Creating handover package at {TARGET_DIR} ...")
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)

    total_bytes = 0
    copied_count = 0

    for src_rel, dst_rel in ITEMS_TO_COPY:
        src_path = ROOT / src_rel
        dst_path = TARGET_DIR / dst_rel
        
        if not src_path.exists():
            print(f"  [WARN] Skipping missing item: {src_rel}")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
            size = src_path.stat().st_size
            total_bytes += size
            copied_count += 1
        elif src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            for f in src_path.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
                    copied_count += 1

    # Write launcher and handover README
    with open(TARGET_DIR / "run_demo.bat", "w", encoding="utf-8") as f:
        f.write(BAT_LAUNCHER)
    with open(TARGET_DIR / "README_HANDOVER.md", "w", encoding="utf-8") as f:
        f.write(README_HANDOVER)

    size_mb = total_bytes / (1024 * 1024)
    size_gb = total_bytes / (1024 * 1024 * 1024)
    print(f"\n[SUCCESS] Handover package successfully created!")
    print(f"  Location: {TARGET_DIR}")
    print(f"  Total Files Copied: {copied_count}")
    print(f"  Total Size: {size_gb:.2f} GB ({size_mb:.1f} MB)")
    print(f"  Fits on 16 GB pendrive: {size_gb < 15.0} (leaves {16.0 - size_gb:.2f} GB free)")

if __name__ == "__main__":
    build_package()
