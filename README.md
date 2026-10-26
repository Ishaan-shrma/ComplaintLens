# ComplaintLens

**Customer Complaint Intelligence & Root-Cause Synthesis Platform**  
*Read-Only Engineering Prototype — Evaluated on Public Proxy Corpora*

---

## 1. System Overview & Engineering Scope

**ComplaintLens** is an end-to-end analytical intelligence system designed to triage customer feedback, detect product and packaging defects, identify negative sentiment signals across diverse review domains, and synthesize evidence-backed root-cause explanations via a grounded, read-only AI agent.

### Technical Stack & Architecture
- **Frontend:** React 18 (`^18.2.0`) + Vite 5 + Tailwind CSS (pure JavaScript JSX; zero TypeScript / `.tsx`).
- **Dashboard Interfaces:** 7 dedicated analytical tabs (`src/components/tabs/`):
  1. `Tab1Corpus.jsx` — Ingestion funnel, deduplication (174,562 purged), and rating distributions.
  2. `Tab2ChampionDeberta.jsx` — DeBERTa-v3 champion architecture, disentangled attention, and OFAT hyperparameter tuning.
  3. `Tab3BenchmarksArchive.jsx` — Multi-model leaderboards, hardware latencies, and threshold curves.
  4. `Tab4LivePredictor.jsx` — Dual-runtime probability triage and live calibrated inference.
  5. `Tab5CrossDomain.jsx` — Domain transfer stress tests across Amazon Food, Yelp Dining, and IMDb Cinema.
  6. `Tab6AiAgent.jsx` — Grounded read-only conversational assistant citing review IDs with guardrails.
  7. `Tab7ErrorDiagnostics.jsx` — 7-tier heuristic failure taxonomy and error rates by token length.
  - *Plus:* Slide-over `LimitationsDrawer.jsx` detailing methodological caveats and system boundaries.
- **Backend:** Python asynchronous FastAPI (`server/app.py`) with thread-safe inference runners and Pydantic validation.
- **Security & Immutability:** Strict read-only database operations; static AST scans of all agent modules verify zero write calls (`to_csv`, `to_sql`, `INSERT`, `UPDATE`, `DELETE`, `DROP`).

---

## 2. Headline Results & Audited Benchmarks

All metrics below are copied directly from the repository's immutable evaluation contracts ([`reports/paired_significance_test.json`](reports/paired_significance_test.json), [`reports/deberta_canonical_test_results.json`](reports/deberta_canonical_test_results.json), and [`reports/model_comparison.csv`](reports/model_comparison.csv)) evaluated across the **canonical held-out test split** ($N = 72,846$; 11,372 negative, 61,474 positive):

| Metric / Attribute | Baseline: DistilBERT (281k) | Champion: DeBERTa-v3-Omni | Source Contract |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | `distilbert-base-uncased` (66M) | `microsoft/deberta-v3-base` (86M) | `model_manifest.json` |
| **Training Corpus** | 281,271 Amazon reviews | 323,461 Multi-domain reviews | `public/data/corpus.json` |
| **Test Accuracy (Raw Text)** | 97.39% | **98.10%** | `reports/paired_significance_test.json` |
| **Macro-F1 Score** | 0.9504 | **0.9646** | `reports/deberta_canonical_test_results.json` |
| **ROC-AUC** | 0.9930 | **0.9957** | `reports/deberta_canonical_test_results.json` |
| **Total Test Errors** | 1,901 | **1,382** (519 fewer errors) | `reports/paired_significance_test.json` |
| **Missed Complaints (FP)** | 980 | **662** (32.4% fewer missed) | `reports/test_predictions_per_row.csv` |
| **False Alarms (FN)** | 921 | **720** (21.8% fewer alarms) | `reports/test_predictions_per_row.csv` |
| **Relative Error Reduction** | *Reference* | **27.3%** (95% CI: [23.97%, 30.51%]) | `reports/paired_significance_test.json` |
| **McNemar Chi-Square** | *Reference* | **$\chi^2 = 199.2012$** ($p < 0.001$) | `reports/paired_significance_test.json` |
| **Platt Calibration ($T^*$)** | $T^* = 1.654$ (ECE $0.0382 \to 0.0121$) | $T^* = 1.350$ (ECE $0.0245 \to 0.0084$) | `public/data/calibration.json` |

---

## 3. Methodological Disclosures & Honest Limitations

1. **Uncontrolled Comparison:** DeBERTa was trained on 323,461 multi-domain reviews with multi-sample dropout and focal loss, while DistilBERT was trained on 281,271 Amazon reviews with standard cross-entropy. This is an end-to-end system comparison, not an isolated single-variable ablation.
2. **Adaptive Test-Set Reuse:** Hyperparameter exploration and temperature calibration were tuned on validation splits from the same source pool; subtle adaptation to benchmark quirks cannot be completely ruled out without external replication.
3. **Replay Training in Cross-Domain Evaluation:** The multi-domain test evaluation (Yelp Dining and IMDb Cinema corpora) included subsets from the same domains within the training replay mix (32,025 reviews total). It evaluates domain-robustness under multi-task exposure, not zero-shot out-of-domain generalization.
4. **Heuristic Error Taxonomy:** The 7-class failure mode classification is computed via deterministic regex patterns and lexical rules (e.g., contrastive clause detection, food idioms, rating inversions), not human-annotated ground truth.
5. **Proxy Data Only:** The system is trained and evaluated entirely on public e-commerce datasets (SNAP Amazon Fine Food Reviews) and synthetic Flipkart templates. It has **not** been trained or validated on proprietary internal consumer reviews of DS Group brands.

---

## 4. Quickstart

### Prerequisites
- **Python:** 3.10–3.12 (Pinned to Python 3.11/3.12; PyTorch lacks official binary wheels for 3.14).
- **Node.js:** v18+ and npm.
- **Operating System:** Windows, macOS, or Linux.

### Setup and Launch
```bash
# 1. Clone repository
git clone https://github.com/<your-org>/DS-Review-AI.git
cd DS-Review-AI

# 2. Initialize environment config (defaults to zero-key offline mode)
cp .env.example .env
# On Windows: copy .env.example .env

# 3. Launch with automated script
# On Windows:
run_demo.bat

# On macOS / Linux:
chmod +x run_demo.sh
./run_demo.sh
```
The launcher starts the FastAPI backend on `http://127.0.0.1:8000` and serves the frontend on `http://localhost:5173`.

---

## 5. Runtime Modes: Replay vs. Live

ComplaintLens supports two operational execution modes determined by local model weight availability:

| Feature | Replay Mode (Zero Downloads) | Live Mode (Weights Present) |
| :--- | :--- | :--- |
| **Requirements** | Cloned repository only (no weights needed) | Checkpoint in `models/deberta_omni_v1/best_model.pt` |
| **UI Experience** | All 7 tabs active, precomputed charts & metrics | All 7 tabs active + real-time inference |
| **Live Prediction** | Disabled with clear HTTP 503 error | Active GPU/CPU PyTorch forward passes |
| **Interactive Examples** | Exact pre-recorded benchmarks labeled `"pre-recorded": true` | Arbitrary custom text classification |
| **Hardware Needed** | Standard laptop CPU; 0 MB model download | NVIDIA GPU recommended (or multi-core CPU) |

To inspect current runtime status, query `GET http://127.0.0.1:8000/health`:
```json
{
  "status": "ok",
  "prototype": "ComplaintLens (Read-Only)",
  "mode": "replay",
  "model_ready": false
}
```

---

## 6. Verification & Automated Test Suites

Independent reproduction is verified through deterministic test commands:

```bash
# 1. Automated Test Suite (42 tests: 32 guardrail tests + 10 prototype verification tests)
pytest -v tests/test_agent.py

# 2. Strict Export Integrity & Recomputation Checks (34 independent checks)
python scripts/verify_export.py

# 3. Frontend AST Literal Linter (enforces 0 hardcoded metrics in src/)
npm run check:literals

# 4. Production Web Bundle Build
npm run build
```

---

## 7. Conversational Agent & Security Perimeter

The AI Review Analyst (`src/agent/`) implements defense-in-depth security policies:
- **Default Offline Operation:** With `LLM_PROVIDER=none` in `.env`, the agent generates deterministic responses using local calculations without any external network traffic or API keys.
- **Provider Adapters:** Supports local offline LLMs via [Ollama](https://ollama.ai) (`LLM_PROVIDER=ollama`) or cloud providers (`LLM_PROVIDER=api`).
- **Citation Requirement:** Every analytical claim must cite specific indexed review IDs.
- **Prompt Injection Defense:** Input prompts attempting SQL injection (`DROP TABLE`), instruction overrides (`ignore previous instructions`), or secret exfiltration are intercepted with HTTP 403 Forbidden.
