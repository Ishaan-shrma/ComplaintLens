from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "paths.config.json"
OUT_DIR = ROOT / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(rel_path: str) -> Path | None:
    if not rel_path or not rel_path.strip():
        return None
    p = ROOT / rel_path
    return p if p.exists() else None


def write_json(filename: str, payload: dict[str, Any]) -> None:
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[EXPORTED] public/data/{filename}")


def export_training_curve(cfg: dict[str, str]) -> None:
    ts_p = resolve_path(cfg.get("trainer_state_json", ""))
    if not ts_p:
        return
    state = json.loads(ts_p.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train_log, val_log = [], []
    for entry in history:
        if "loss" in entry or "train_loss" in entry:
            raw_l = entry.get("train_loss", entry.get("loss"))
            if raw_l is not None and "eval_loss" not in entry:
                train_log.append(
                    {
                        "step": entry.get("step"),
                        "epoch": round(float(entry.get("epoch", 0)), 2),
                        "train_loss": round(float(raw_l), 4),
                    }
                )
        if "eval_loss" in entry:
            val_log.append(
                {
                    "step": entry.get("step"),
                    "epoch": round(float(entry.get("epoch", 0)), 2),
                    "eval_loss": round(float(entry["eval_loss"]), 4),
                    "eval_macro_f1": round(float(entry.get("eval_macro_f1", 0)), 4),
                }
            )
    write_json(
        "training_curve.json",
        {
            "source_files": [cfg.get("trainer_state_json")],
            "generated_at": now_iso(),
            "split": "train (n = 281,271) & validation (n = 9,997)",
            "n": 281271,
            "model": "distilbert-base-uncased (281k train)",
            "train_log": train_log,
            "val_log": val_log,
        },
    )


def export_passthrough(cfg_key: str, out_name: str, cfg: dict[str, str]) -> None:
    src = resolve_path(cfg.get(cfg_key, ""))
    if not src:
        target = OUT_DIR / out_name
        if target.exists():
            target.unlink()
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    data.setdefault("source_files", [cfg.get(cfg_key)])
    data.setdefault("generated_at", now_iso())
    write_json(out_name, data)


def export_guardrails(cfg: dict[str, str]) -> None:
    log_p = resolve_path(cfg.get("pytest_log_txt", ""))
    if not log_p:
        return
    raw_text = log_p.read_text(encoding="utf-8")
    pattern = re.compile(r"(tests/test_agent\.py::[\w\[\]\-]+)\s+(PASSED|FAILED|SKIPPED)")
    tests = []
    for match in pattern.finditer(raw_text):
        full_name, status = match.group(1), match.group(2)
        short = full_name.split("::")[-1]
        cat = (
            "blocked_threat"
            if "block" in short.lower()
            else "allowed_analytical"
            if "allow" in short.lower()
            else "static_ast_scan"
        )
        tests.append({"test_id": full_name, "name": short, "status": status, "category": cat})

    cat_counts: dict[str, int] = {}
    for t in tests:
        cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1

    write_json(
        "guardrails.json",
        {
            "source_files": [cfg.get("pytest_log_txt")],
            "generated_at": now_iso(),
            "recorded_log_date": datetime.fromtimestamp(
                log_p.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
            "split": "pytest security & AST verification suite",
            "n": len(tests),
            "model": "Deterministic Guardrail Layer + Regex PII Masking",
            "passed_count": sum(1 for t in tests if t["status"] == "PASSED"),
            "failed_count": sum(1 for t in tests if t["status"] == "FAILED"),
            "category_counts": cat_counts,
            "tests": tests,
        },
    )


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    export_passthrough("cleaning_summary_json", "corpus.json", cfg)
    export_passthrough("final_test_results_json", "final_model.json", cfg)
    export_passthrough("calibration_json", "calibration.json", cfg)
    export_training_curve(cfg)
    export_passthrough("model_comparison_json", "model_comparison.json", cfg)
    export_passthrough("distilbert_results_json", "ablation.json", cfg)
    export_passthrough("deberta_results_json", "deberta.json", cfg)
    export_passthrough("retrieval_benchmark_json", "retrieval.json", cfg)
    export_passthrough("error_taxonomy_json", "error_taxonomy.json", cfg)
    export_passthrough("search_demo_json", "search_demo.json", cfg)
    export_passthrough("recorded_predictions_json", "recorded_predictions.json", cfg)
    export_passthrough("recorded_agent_json", "recorded_agent.json", cfg)
    export_guardrails(cfg)


if __name__ == "__main__":
    main()
