# Review Intelligence Prototype

> **Read-Only Engineering Prototype — Not Validated for Production Deployment.**  
> Built as part of a 45-day AI/ML Engineering internship at DS Group. All interfaces, analytical endpoints, and conversational capabilities are strictly read-only and operate without write or database mutation handles.

---

## 1. System Overview & Limitations

The **Review Intelligence Prototype** is an analytical evaluation system designed to process e-commerce customer feedback, identify product defects, evaluate sentiment signals across multi-domain corpora, and provide evidence-grounded responses to analytical queries.

### System Limitations & Prototype Scope
- **Not Validated for Production:** This repository is an educational and research prototype. Model calibration, error distributions, and inference latency were measured under experimental constraints and have not been certified for live production traffic.
- **Corpus Boundaries:** Primary evaluations derive from the SNAP Amazon Fine Food Reviews dataset (1999–2012; 393,890 deduplicated reviews) and the Universal Omni Corpus (453,878 multi-platform reviews). Star ratings are approximate proxies for true customer sentiment.
- **Read-Only Enforcement:** The architecture strictly prohibits data alteration. A static AST scan of all 5 agent modules confirms zero write calls (`to_csv`, `to_sql`, `INSERT`, `UPDATE`, `DELETE`, `DROP`).

---

## 2. Architecture & Headline Results

### Dual-Model Architecture
The prototype compares two transformer architectures evaluated on the same identical benchmark set:

| Dimension | Primary Model: DeBERTa-v3-Omni | Baseline Comparison: DistilBERT |
| :--- | :--- | :--- |
| **Base Backbone** | `microsoft/deberta-v3-base` (86M params, disentangled attention) | `distilbert-base-uncased` (66M params) |
| **Pooling Mechanism** | Attention-Masked Mean + Max Pooling concatenation | Standard `[CLS]` token pooling |
| **Classification Head** | 5-Branch Multi-Sample Dropout ($p \in [0.1, 0.5]$) | Single linear classification head |
| **Training Objective** | Focal Loss ($\gamma = 2.0, \alpha = 0.25$) + Class Weighting | Standard Cross-Entropy Loss |
| **Calibration Temperature** | $T = 1.35$ | $T = 1.654$ |
| **Training Corpus Size** | $n = 323,461$ (Multi-platform Omni Corpus) | $n = 281,271$ (Amazon Fine Food train slice) |

### Headline Evaluation Results ($n = 72,846$ Test Set)
Under identical raw text input evaluation on the held-out test split ($n = 72,846$):
- **DeBERTa-v3-Omni:** 1,382 classification errors (98.10% accuracy, 0.9646 Macro-F1, ROC-AUC 0.9957)
- **DistilBERT Baseline:** 1,901 classification errors (97.39% accuracy, 0.9504 Macro-F1, ROC-AUC 0.9930)
- **Error Reduction:** A reduction of 519 errors (27.3% relative error reduction) associated with the architectural upgrades.

### Critical Methodological Caveats
1. **Bundled Differences:** The performance difference reflects multiple simultaneous modifications (backbone architecture, pooling, dropout ensemble head, focal loss, and expanded training corpus). It is an aggregate system comparison, not an isolated single-variable ablation.
2. **Adaptive Test-Set Reuse:** Hyperparameter exploration and threshold tuning were conducted on validation splits derived from the same source pool; subtle overfitting to benchmark quirks cannot be ruled out without subsequent external replication.
3. **Multi-Platform Omni Corpus Composition:** The Omni corpus (453,878 reviews) combines **three real-world review sources** (Amazon Fine Food, Yelp Restaurant, IMDb Movie reviews) with **one synthetic/adversarial slice** (Flipkart reviews featuring code-mixed Hinglish, informal slang, and high-frequency sarcastic sarcasm) to test model robustness under out-of-domain linguistic stress.

---

## 3. Quickstart (3 Steps)

### Prerequisites
- Python 3.10–3.12 (Pinned to **Python 3.11.9** via `.python-version`; PyTorch wheels are not yet officially released for Python 3.14).
- Node.js (v18+) and npm.
- Git.

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/DS-Review-AI.git
cd DS-Review-AI
```

### Step 2: Configure Environment
Copy the example configuration to initialize the local `.env`:
```bash
cp .env.example .env
# Windows: copy .env.example .env
```
*(By default, `LLM_PROVIDER=none`, enabling zero-dependency deterministic operation without external API keys).*

### Step 3: Launch Prototype
Run the automated launcher:
- **Windows:**
  ```cmd
  run_demo.bat
  ```
- **macOS / Linux:**
  ```bash
  chmod +x run_demo.sh
  ./run_demo.sh
  ```

The launcher will verify `.venv`, install requirements, install npm dependencies, launch the FastAPI server on `http://127.0.0.1:8000`, and start the Vite UI on `http://localhost:5173`.

---

## 4. Runtime Modes

The prototype server (`server/app.py`) dynamically detects available local assets and starts in one of two execution modes:

### Inference Modes (`mode`)
- **`live` Mode (Weights Present):** Activated when trained model checkpoints are located in `models/`. The server runs real PyTorch tensor forward passes and returns live calibrated probabilities.
- **`replay` Mode (Weights Absent):** If model weights are missing (e.g., in a fresh repository clone before asset download), the server starts safely in `replay` mode.
  - **Live prediction for arbitrary new text is strictly DISABLED** with an explicit HTTP 503 response.
  - Only exact pre-recorded benchmark examples from `public/data/recorded_predictions.json` may be displayed, with each output marked `"pre-recorded": true`.
  - The server **never returns simulated or invented probabilities** for arbitrary input.

### Data Modes (`data_mode`)
- **`full` Data Mode:** Active when the complete cleaned corpus `data/processed/reviews_cleaned.csv` is present.
- **`sample` Data Mode:** Active when operating from the committed 5,000-review sample `data/sample/sample_reviews_5k.csv`. In sample mode, all AI agent answers and KPI cards explicitly indicate:  
  `"(computed on a 5,000-review sample; differs from full-corpus figures)"`.

The active status of both modes is visible via `GET http://127.0.0.1:8000/health`.

---

## 5. Hardware Notes

- **GPU Acceleration:** CUDA acceleration is detected automatically. If an NVIDIA GPU with compatible CUDA drivers is present, inference runs on CUDA.
- **CPU Fallback:** If no GPU is detected, the prototype executes entirely on CPU.
- **Latency Expectation:** CPU forward passes are significantly slower than GPU execution (typically 150–400ms per inference on CPU versus 15–25ms on GPU). Always time inference latency locally on your specific hardware before quoting speed benchmarks.

---

## 6. Large Files Not in This Repository

To comply with Git repository hygiene standards (strictly zero files over 50 MB committed), large binary weights, indices, and raw data are excluded via `.gitignore`:

| Excluded Asset | Expected Location | Size | Description |
| :--- | :--- | :--- | :--- |
| **DeBERTa Checkpoint** | `models/deberta_omni_v1/best_model.pt` | ~500 MB | PyTorch weights for primary classifier |
| **DistilBERT Checkpoint** | `models/sentiment_distilbert/model.safetensors` | ~268 MB | Baseline classifier weights |
| **FAISS Vector Index** | `models/faiss/faiss_index_with_ids.bin` | ~150 MB | Semantic retrieval index |
| **Embeddings Array** | `models/embeddings.npy` | 146.5 MB | Precomputed sentence embeddings |
| **Cleaned Corpus** | `data/processed/reviews_cleaned.csv` | ~280 MB | Full 393k Amazon review corpus |
| **Raw Kaggle Data** | `data/raw/Reviews.csv` | ~300 MB | Original raw Kaggle export |

### Asset Downloader
To download model weights and full datasets into their expected locations:
```bash
python scripts/download_assets.py
```
*(Pass `--verify-only` to check SHA-256 integrity of existing local files without downloading. If downloading from a private Hugging Face repository, set `HF_TOKEN` in your environment).*

---

## 7. Enable Chatbot Answers (Optional)

The AI Review Analyst chat interface operates with three flexible LLM provider adapters:

### a. Default Deterministic Mode (No Key Required)
Set `LLM_PROVIDER=none` in `.env` (the repository default). The agent answers all valid analytical questions using deterministic template responses constructed strictly from approved tool outputs. No external calls or API keys are required.

### b. Ollama (Local Inference, No External Calls)
1. Install [Ollama](https://ollama.ai) on your system.
2. Pull a local model: `ollama pull llama3` (or `mistral`, `qwen2.5`).
3. Configure `.env`:
   ```bash
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3
   LLM_BASE_URL=http://localhost:11434
   ```
4. Restart the server. In this mode, zero network traffic leaves your local workstation.

### c. External API Providers (Groq, OpenAI, etc.)
1. Copy `.env.example` to `.env`.
2. Configure provider settings:
   ```bash
   LLM_PROVIDER=api
   LLM_BASE_URL=https://api.groq.com/openai/v1   # Or provider base URL
   LLM_MODEL=llama-3.1-8b-instant               # Check provider console for current active names
   LLM_API_KEY=your_actual_api_key_here
   ```
3. Set a strict spending limit in your provider console.
4. Restart the server. Verify provider readiness via `/health`, confirming `key_present: true` and `reachable: true`. *(Always check provider model names and pricing before querying, as catalog availability changes).*

### d. Data Privacy Notice
- In `api` mode, the user's prompt and retrieved review snippets are transmitted to the external API provider.
- In `none` and `ollama` modes, **zero data leaves your local machine**.

### e. Troubleshooting & Fallback
- If an API key is missing, invalid (HTTP 401/403), timed out, or rate-limited, the agent **automatically catches the error, redacts any error traces, and gracefully falls back to deterministic template answers**.
- To manually revert to offline mode at any point, set `LLM_PROVIDER=none`.

### f. Security Best Practices
- Never commit `.env` to Git. (The `.gitignore` strictly ignores `.env` and `.env.*`).
- If an API key is accidentally exposed, immediately revoke it in the provider console and generate a replacement.

---

## 8. Chat UI Behaviour & Guardrails

The conversational interface implements strict read-only analytical guardrails:
- **Response Labeling:** Every answer displays whether it was generated by a language model (`"language model: <name>"`) or generated deterministically (`"template answer (answered without a language model)"`).
- **Evidence Citation:** Answers cite specific Review IDs (e.g., `Review #1042`) linking directly to source text.
- **Tool Whitelist:** The agent can only invoke 4 read-only tools: `search_reviews`, `get_review_kpi`, `predict_sentiment`, and `explain_result`. The model cannot write SQL, execute shell commands, or perform mutations.
- **Untrusted Context Isolation:** Retrieved review text is passed in `<untrusted_review_context>` XML blocks; embedded injection instructions inside customer reviews are ignored.
- **Strict Number Grounding:** Every numerical metric presented in an answer must strictly match an approved tool output.

---

## 9. How to Verify (The 5 Verification Commands)

Run these five commands locally to independently verify all system invariants:

```bash
# 1. Full Guardrail & Agent Test Suite (42 test cases)
pytest -v tests/test_agent.py

# 2. Strict Export & Independent Metric Recomputation (34 checks)
python scripts/verify_export.py

# 3. AST Literal Checker (verifies 0 hardcoded metric literals in src/)
npm run check:literals

# 4. Vite Production Build
npm run build

# 5. Git Repository Size Check (verifies 0 files > 50 MB)
python scripts/check_repo_size.py
```

---

## 10. For Reviewers

- **Evidence Tables & Data Contracts:** All baseline metrics, confusion matrices, error taxonomies, and multi-domain test results are tracked in versioned JSON data contracts in `public/data/` and documented in `reports/`.
- **Independent vs. Self-Consistency Verification:**  
  - *Independent checks* in `scripts/verify_export.py` re-derive metrics directly from raw per-row test predictions (`test_predictions_per_row.csv`) and length slices.  
  - *Self-consistency checks* verify that confusion matrices, platform sums, and error taxonomies sum symmetrically to total dataset counts ($n = 72,846$ and $N = 453,878$).
- **Reviewer Notice:** *Nothing in this repository is approved for operational use until reviewed and validated by the project mentor.*
