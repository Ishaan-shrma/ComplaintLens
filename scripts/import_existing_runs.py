"""
scripts/import_existing_runs.py
================================
One-time migration: reads all historical training runs from the notebook
artifact JSON files and loads them into the SQLite experiment tracker
(runs/experiments.db) via RunLogger.

Run from the project root:
    python scripts/import_existing_runs.py

Safe to re-run — it checks for duplicate run names before inserting.
"""

import json
import sys
from pathlib import Path

# ── Make src/ importable ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.logger import RunLogger

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS = PROJECT_ROOT / "notebooks" / "artifacts"
DISTILBERT_RUNS   = ARTIFACTS / "distilbert_results.json"   # 15 ablation runs
SEED2_RUNS        = ARTIFACTS / "seed2_results.json"        # 2 seed-stability runs
FINAL_TEST        = ARTIFACTS / "final_test_results.json"   # official test-set scores
REPORTS_DIR       = PROJECT_ROOT / "reports"

# ── Existing saved images (if present) ───────────────────────────────────────
EXISTING_IMAGES = {
    # map run_name → (cm_path, curve_path) for any images already on disk
    "full281k_lr2e-5_bs16_ep2": (
        str(REPORTS_DIR / "model_refinement_comparison.png"),
        None,
    ),
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def already_imported(logger: RunLogger, run_name: str) -> bool:
    """Return True if a run with this name already exists in the DB."""
    all_runs = logger.get_all_runs()
    return any(r["run_name"] == run_name for r in all_runs)


def import_ablation_runs(logger: RunLogger, runs: list, source_tag: str):
    """Import ablation/grid-search runs that have val metrics but no test metrics."""
    imported = 0
    skipped  = 0

    for r in runs:
        run_name = r["run"]

        if already_imported(logger, run_name):
            print(f"  ⏭  SKIP  {run_name}  (already in DB)")
            skipped += 1
            continue

        # Build per-epoch loss list
        val_losses = r.get("val_loss_by_epoch", [])
        epochs_run = r.get("epochs", len(val_losses))

        run_id = logger.start_run(
            model_name   = "distilbert-base-uncased",
            run_name     = run_name,
            epochs       = epochs_run,
            lr           = r.get("lr", 2e-5),
            batch_size   = r.get("batch", 16),
            seed         = r.get("seed", 42),
            train_size   = 50_000,   # ablation runs used 50k subset
            fp16         = True,
            max_length   = 128,
            weight_decay = 0.01,
            notes        = f"[{source_tag}] {r.get('change', '')}",
        )

        # Log per-epoch val loss (no train_loss stored in the JSONs)
        for ep_idx, v_loss in enumerate(val_losses, start=1):
            logger.log_epoch(
                run_id,
                epoch        = ep_idx,
                val_loss     = v_loss,
                val_macro_f1 = r.get("val_macro_f1") if ep_idx == epochs_run else None,
                val_neg_recall = r.get("neg_recall") if ep_idx == epochs_run else None,
            )

        # Resolve artefact image paths
        cm_path, curve_path = EXISTING_IMAGES.get(run_name, (None, None))

        # Checkpoint path for the two saved models
        ckpt = None
        if run_name == "full281k_lr2e-5_bs16_ep2":
            ckpt = str(ARTIFACTS / "distilbert_full281k")
        elif run_name == "run1":
            ckpt = str(ARTIFACTS / "distilbert_final")

        logger.finish_run(
            run_id,
            val_macro_f1     = r.get("val_macro_f1"),
            val_accuracy     = r.get("val_acc"),
            val_neg_recall   = r.get("neg_recall"),
            val_neg_f1       = r.get("neg_f1"),
            best_epoch       = r.get("best_epoch"),
            training_minutes = r.get("minutes"),
            cm_image_path    = cm_path,
            curve_image_path = curve_path,
            checkpoint_path  = ckpt,
        )

        print(f"  ✅ IMPORTED  {run_name}  (val_macro_f1={r.get('val_macro_f1')})")
        imported += 1

    return imported, skipped


def import_final_test_results(logger: RunLogger, final: dict):
    """
    Import the three officially evaluated models from final_test_results.json.
    These have full test-set metrics. We upsert them by run_name.
    """
    model_map = {
        "svm_tfidf_linearsvc_C0.5_cwNone": {
            "run_name"   : "svm_tfidf_C0.5_cwNone",
            "model_name" : "LinearSVC-TF-IDF",
            "lr"         : 0.5,      # SVM C parameter (reused lr field)
            "epochs"     : 1,
            "batch_size" : 0,
            "train_size" : 291_276,
            "notes"      : "Tuned SVM baseline. C=0.5, TF-IDF, no class weight.",
            "ckpt"       : str(PROJECT_ROOT / "models" / "svm_pipeline.joblib"),
        },
        "distilbert_50k": {
            "run_name"   : "distilbert_50k_diagnostic",
            "model_name" : "distilbert-base-uncased",
            "lr"         : 2e-5,
            "epochs"     : 2,
            "batch_size" : 16,
            "train_size" : 50_000,
            "notes"      : "50k diagnostic ablation. Proved naive 4x class weighting degrades Macro-F1.",
            "ckpt"       : None,
        },
        "distilbert_281k_FINAL": {
            "run_name"   : "distilbert_281k_FINAL",
            "model_name" : "distilbert-base-uncased",
            "lr"         : 2e-5,
            "epochs"     : 2,
            "batch_size" : 16,
            "train_size" : 281_271,
            "notes"      : "Production v1. Full 281k training, T=1.654 calibration. Test ECE=0.0094.",
            "ckpt"       : str(ARTIFACTS / "distilbert_full281k"),
        },
    }

    hparams = final.get("hyperparameters_final", {})
    imported = 0
    skipped  = 0

    for key, meta in model_map.items():
        data = final.get(key)
        if not data:
            print(f"  ⚠  Key '{key}' not found in final_test_results.json — skipping")
            continue

        run_name = meta["run_name"]
        if already_imported(logger, run_name):
            print(f"  ⏭  SKIP  {run_name}  (already in DB)")
            skipped += 1
            continue

        # Temperature only exists for the final DistilBERT model
        temp_T = None
        if key == "distilbert_281k_FINAL":
            import numpy as np
            t_path = ARTIFACTS / "temperature_T.npy"
            if t_path.exists():
                temp_T = float(np.load(str(t_path)).item())

        run_id = logger.start_run(
            model_name   = meta["model_name"],
            run_name     = run_name,
            epochs       = meta["epochs"],
            lr           = meta["lr"],
            batch_size   = meta["batch_size"],
            seed         = hparams.get("seed", 42),
            train_size   = meta["train_size"],
            fp16         = bool(hparams.get("fp16", True)),
            max_length   = hparams.get("max_length", 128),
            weight_decay = hparams.get("weight_decay", 0.01),
            temperature_T = temp_T,
            notes        = meta["notes"],
        )

        logger.finish_run(
            run_id,
            test_macro_f1     = data.get("macro_f1"),
            test_neg_recall   = data.get("neg_recall"),
            test_roc_auc      = data.get("roc_auc"),
            test_accuracy     = data.get("accuracy"),
            test_neg_precision = data.get("neg_precision"),
            test_neg_f1       = data.get("neg_f1"),
            test_pos_f1       = data.get("pos_f1"),
            test_TN           = data.get("TN"),
            test_FP           = data.get("FP"),
            test_FN           = data.get("FN"),
            test_TP           = data.get("TP"),
            temperature_T     = temp_T,
            checkpoint_path   = meta["ckpt"],
            cm_image_path     = str(REPORTS_DIR / "model_refinement_comparison.png")
                                if key == "distilbert_281k_FINAL" else None,
        )

        print(f"  ✅ IMPORTED  {run_name}  (test_macro_f1={data.get('macro_f1')})")
        imported += 1

    return imported, skipped


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Review Intelligence — Experiment Run Importer")
    print("=" * 60)

    logger = RunLogger()
    print(f"\n  DB path: {logger.db_path}\n")

    total_imported = 0
    total_skipped  = 0

    # 1. Ablation runs (distilbert_results.json — 15 runs)
    print("── distilbert_results.json ──────────────────────────────")
    ablation_runs = load_json(DISTILBERT_RUNS)
    imp, skp = import_ablation_runs(logger, ablation_runs, "ablation")
    total_imported += imp
    total_skipped  += skp

    # 2. Seed-stability runs (seed2_results.json — 2 runs)
    print("\n── seed2_results.json ───────────────────────────────────")
    seed_runs = load_json(SEED2_RUNS)
    imp, skp = import_ablation_runs(logger, seed_runs, "seed-stability")
    total_imported += imp
    total_skipped  += skp

    # 3. Official test-set results (final_test_results.json — 3 models)
    print("\n── final_test_results.json ──────────────────────────────")
    final_data = load_json(FINAL_TEST)
    imp, skp = import_final_test_results(logger, final_data)
    total_imported += imp
    total_skipped  += skp

    # ── Summary ──────────────────────────────────────────────────────────────
    all_runs = logger.get_all_runs()
    print(f"\n{'=' * 60}")
    print(f"  Done.  Imported: {total_imported}  |  Skipped: {total_skipped}")
    print(f"  Total runs now in DB: {len(all_runs)}")
    print(f"{'=' * 60}\n")

    # Quick preview table
    print(f"  {'Run Name':<40} {'Val F1':>7}  {'Test F1':>8}")
    print(f"  {'-'*40} {'-'*7}  {'-'*8}")
    for r in reversed(all_runs):
        val  = f"{r['val_macro_f1']:.4f}"  if r["val_macro_f1"]  else "  --   "
        test = f"{r['test_macro_f1']:.4f}" if r["test_macro_f1"] else "  --   "
        print(f"  {r['run_name']:<40} {val:>7}  {test:>8}")
    print()


if __name__ == "__main__":
    main()
