"""
src/training/logger.py
======================
RunLogger — SQLite-backed experiment tracker for the Review Intelligence platform.

Every training run (DistilBERT, DeBERTa, SVM, etc.) is stored as a single row
with all hyperparameters, metrics, and paths to saved artefacts (confusion matrix
image, loss-curve image, model checkpoint).

Usage (inside a training script)
---------------------------------
    from src.training.logger import RunLogger

    logger = RunLogger()                    # opens / creates runs/experiments.db
    run_id = logger.start_run(
        model_name   = "deberta-v3-base",
        epochs       = 3,
        lr           = 2e-5,
        batch_size   = 16,
        notes        = "LLRD alpha=0.90, Focal gamma=2.0",
    )

    # Inside your epoch loop:
    logger.log_epoch(run_id, epoch=1, train_loss=0.12, val_loss=0.09,
                     val_macro_f1=0.95, val_neg_recall=0.91)

    # After evaluation on the test set:
    logger.finish_run(
        run_id,
        test_macro_f1    = 0.9612,
        test_neg_recall  = 0.9210,
        test_roc_auc     = 0.9941,
        test_ece         = 0.0082,
        temperature_T    = 1.421,
        threshold        = 0.50,
        cm_image_path    = "runs/images/run_007_cm.png",
        curve_image_path = "runs/images/run_007_loss.png",
        checkpoint_path  = "models/deberta_omni_v1/",
    )
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Default DB location (project root / runs / experiments.db) ──────────────
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "runs" / "experiments.db"


class RunLogger:
    """Lightweight SQLite-backed ML experiment tracker."""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    _CREATE_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id            TEXT PRIMARY KEY,
        run_name          TEXT,
        model_name        TEXT,
        created_at        TEXT,
        finished_at       TEXT,
        status            TEXT DEFAULT 'running',

        -- Hyperparameters
        epochs            INTEGER,
        lr                REAL,
        batch_size        INTEGER,
        weight_decay      REAL,
        warmup_ratio      REAL,
        focal_gamma       REAL,
        label_smoothing   REAL,
        llrd_alpha        REAL,
        max_length        INTEGER,
        fp16              INTEGER,  -- 0/1 boolean
        seed              INTEGER,
        class_weight      TEXT,     -- e.g. "balanced" or null
        temperature_T     REAL,
        threshold         REAL,
        train_size        INTEGER,
        extra_params      TEXT,     -- JSON blob for anything else

        -- Val metrics (best epoch)
        val_macro_f1      REAL,
        val_accuracy      REAL,
        val_neg_recall    REAL,
        val_neg_f1        REAL,
        best_epoch        INTEGER,
        training_minutes  REAL,

        -- Test-set metrics (set once, never changed)
        test_macro_f1     REAL,
        test_neg_recall   REAL,
        test_roc_auc      REAL,
        test_ece          REAL,
        test_accuracy     REAL,
        test_neg_precision REAL,
        test_neg_f1       REAL,
        test_pos_f1       REAL,
        test_TN           INTEGER,
        test_FP           INTEGER,
        test_FN           INTEGER,
        test_TP           INTEGER,

        -- Artefacts
        cm_image_path     TEXT,
        curve_image_path  TEXT,
        checkpoint_path   TEXT,

        notes             TEXT
    )
    """

    _CREATE_EPOCHS = """
    CREATE TABLE IF NOT EXISTS epoch_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT REFERENCES runs(run_id),
        epoch       INTEGER,
        train_loss  REAL,
        val_loss    REAL,
        val_macro_f1  REAL,
        val_neg_recall REAL,
        logged_at   TEXT
    )
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self._CREATE_RUNS)
            conn.execute(self._CREATE_EPOCHS)
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_run(
        self,
        *,
        model_name: str,
        epochs: int,
        lr: float,
        batch_size: int,
        run_name: Optional[str] = None,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        focal_gamma: Optional[float] = None,
        label_smoothing: Optional[float] = None,
        llrd_alpha: Optional[float] = None,
        max_length: int = 128,
        fp16: bool = True,
        seed: int = 42,
        class_weight: Optional[str] = None,
        temperature_T: Optional[float] = None,
        threshold: float = 0.50,
        train_size: Optional[int] = None,
        notes: str = "",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new run. Returns the unique run_id."""
        run_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat(timespec="seconds")

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (
                    run_id, run_name, model_name, created_at, status,
                    epochs, lr, batch_size, weight_decay, warmup_ratio,
                    focal_gamma, label_smoothing, llrd_alpha, max_length,
                    fp16, seed, class_weight, temperature_T, threshold,
                    train_size, notes, extra_params
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_name or f"{model_name}_{now[:10]}",
                    model_name,
                    now,
                    epochs, lr, batch_size, weight_decay, warmup_ratio,
                    focal_gamma, label_smoothing, llrd_alpha, max_length,
                    int(fp16), seed, class_weight, temperature_T, threshold,
                    train_size, notes,
                    json.dumps(extra_params) if extra_params else None,
                ),
            )
            conn.commit()
        return run_id

    def log_epoch(
        self,
        run_id: str,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_macro_f1: Optional[float] = None,
        val_neg_recall: Optional[float] = None,
    ):
        """Append one epoch record. Call inside your training loop."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO epoch_logs
                   (run_id, epoch, train_loss, val_loss, val_macro_f1, val_neg_recall, logged_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, epoch, train_loss, val_loss, val_macro_f1,
                 val_neg_recall, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()

    def finish_run(
        self,
        run_id: str,
        *,
        val_macro_f1: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        val_neg_recall: Optional[float] = None,
        val_neg_f1: Optional[float] = None,
        best_epoch: Optional[int] = None,
        training_minutes: Optional[float] = None,
        test_macro_f1: Optional[float] = None,
        test_neg_recall: Optional[float] = None,
        test_roc_auc: Optional[float] = None,
        test_ece: Optional[float] = None,
        test_accuracy: Optional[float] = None,
        test_neg_precision: Optional[float] = None,
        test_neg_f1: Optional[float] = None,
        test_pos_f1: Optional[float] = None,
        test_TN: Optional[int] = None,
        test_FP: Optional[int] = None,
        test_FN: Optional[int] = None,
        test_TP: Optional[int] = None,
        temperature_T: Optional[float] = None,
        cm_image_path: Optional[str] = None,
        curve_image_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Mark a run complete and write all final metrics."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET
                    status='done', finished_at=?,
                    val_macro_f1=?, val_accuracy=?, val_neg_recall=?, val_neg_f1=?,
                    best_epoch=?, training_minutes=?,
                    test_macro_f1=?, test_neg_recall=?, test_roc_auc=?, test_ece=?,
                    test_accuracy=?, test_neg_precision=?, test_neg_f1=?, test_pos_f1=?,
                    test_TN=?, test_FP=?, test_FN=?, test_TP=?,
                    temperature_T=COALESCE(?, temperature_T),
                    cm_image_path=?, curve_image_path=?, checkpoint_path=?
                WHERE run_id=?""",
                (
                    now,
                    val_macro_f1, val_accuracy, val_neg_recall, val_neg_f1,
                    best_epoch, training_minutes,
                    test_macro_f1, test_neg_recall, test_roc_auc, test_ece,
                    test_accuracy, test_neg_precision, test_neg_f1, test_pos_f1,
                    test_TN, test_FP, test_FN, test_TP,
                    temperature_T,
                    cm_image_path, curve_image_path, checkpoint_path,
                    run_id,
                ),
            )
            conn.commit()

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Return all runs as a list of dicts (for Streamlit dataframes)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single run by run_id."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_epochs(self, run_id: str) -> List[Dict[str, Any]]:
        """Return per-epoch logs for a run (used by the training-curve slider)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM epoch_logs WHERE run_id=? ORDER BY epoch ASC", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: str):
        """Hard-delete a run and its epoch logs. Use with caution."""
        with self._connect() as conn:
            conn.execute("DELETE FROM epoch_logs WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
            conn.commit()
