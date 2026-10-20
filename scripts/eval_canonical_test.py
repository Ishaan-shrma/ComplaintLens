"""
scripts/eval_canonical_test.py
==============================
Rigorous Independent Evaluation of Champion DeBERTa-v3-Omni
on the 72,846 Canonical Held-Out Test Reviews (notebooks/artifacts/test_72846.pkl).

Produces the definitive benchmark metrics:
  - Accuracy
  - Macro-F1
  - Negative Recall (Minority class sensitivity)
  - Negative Precision
  - ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)
  - Direct head-to-head comparison against DistilBERT baseline
"""

import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.deberta_omni import DeBERTaOmniClassifier
from src.data.normalizer import batch_normalize

MODEL_DIR = ROOT / "models" / "deberta_omni_v1"
TEST_PKL  = ROOT / "notebooks" / "artifacts" / "test_72846.pkl"
OUT_JSON  = ROOT / "reports" / "deberta_canonical_test_results.json"
MAX_LENGTH = 128
BATCH_SIZE = 48


def evaluate_test():
    print("=" * 70)
    print("  CANONICAL TEST SET BENCHMARK EVALUATION (N = 72,846)")
    print(f"  Model Checkpoint: {MODEL_DIR / 'best_model.pt'}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("\n[1/4] Loading canonical held-out test split...")
    with open(TEST_PKL, "rb") as f:
        test_df = pickle.load(f)

    test_texts  = test_df["text"].tolist() if "text" in test_df else test_df["Text"].tolist()
    test_labels = test_df["label"].astype(int).tolist()
    print(f"Loaded {len(test_texts):,} reviews. Negative: {test_labels.count(0):,} | Positive: {test_labels.count(1):,}")

    print("\n[2/4] Applying Wild-Web Text Normalizer...")
    t0 = time.time()
    test_texts = batch_normalize(test_texts)
    print(f"Normalized {len(test_texts):,} reviews in {time.time()-t0:.1f}s.")

    print("\n[3/4] Tokenizing test dataset...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR) if (MODEL_DIR / "tokenizer.json").exists() else "microsoft/deberta-v3-base")
    enc = tokenizer(
        [str(t) for t in test_texts],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    test_ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    print("\n[4/4] Loading Champion Model weights & running inference...")
    model = DeBERTaOmniClassifier(model_name="microsoft/deberta-v3-base", num_labels=2)
    weights_path = MODEL_DIR / "best_model.pt"
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_preds = []
    all_probs = []
    all_targets = []

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    start_eval = time.time()
    with torch.no_grad():
        for b_in, b_att, b_lbl in tqdm(test_loader, desc="Evaluating Test Set", unit="batch"):
            b_in  = b_in.to(device, non_blocking=True)
            b_att = b_att.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(b_in, b_att)
                probs = torch.softmax(out["logits"].float(), dim=-1).cpu().numpy()

            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])
            all_targets.extend(b_lbl.numpy())

    eval_time = (time.time() - start_eval) / 60.0

    all_preds   = np.array(all_preds)
    all_probs   = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Metrics calculation
    acc        = accuracy_score(all_targets, all_preds)
    macro_f1   = f1_score(all_targets, all_preds, average="macro")
    neg_recall = recall_score(all_targets, all_preds, pos_label=0)
    neg_prec   = precision_score(all_targets, all_preds, pos_label=0)
    neg_f1     = f1_score(all_targets, all_preds, pos_label=0)
    pos_recall = recall_score(all_targets, all_preds, pos_label=1)
    pos_prec   = precision_score(all_targets, all_preds, pos_label=1)
    pos_f1     = f1_score(all_targets, all_preds, pos_label=1)
    roc_auc    = roc_auc_score(all_targets, all_probs)
    cm         = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 70)
    print("  DEFINITIVE BENCHMARK RESULTS — CANONICAL TEST SET (72,846 REVIEWS)")
    print("=" * 70)
    print(f"  Accuracy           : {acc * 100:.2f}%")
    print(f"  Macro-F1           : {macro_f1:.4f}")
    print(f"  Negative Recall    : {neg_recall * 100:.2f}%  (TN={tn:,} / Actual Negatives={tn+fp:,})")
    print(f"  Negative Precision : {neg_prec * 100:.2f}%  (FP={fp:,})")
    print(f"  Negative F1        : {neg_f1:.4f}")
    print(f"  Positive Recall    : {pos_recall * 100:.2f}%  (TP={tp:,} / Actual Positives={tp+fn:,})")
    print(f"  Positive Precision : {pos_prec * 100:.2f}%  (FN={fn:,})")
    print(f"  Positive F1        : {pos_f1:.4f}")
    print(f"  ROC-AUC Score      : {roc_auc:.4f}")
    print(f"  Inference Duration : {eval_time:.1f} minutes ({len(test_texts)/((eval_time*60) or 1):.1f} reviews/sec)")
    print("=" * 70)

    print("\n  CONFUSION MATRIX:")
    print(f"                 Predicted Negative    Predicted Positive")
    print(f"  Actual Neg   : {tn:<20} {fp:<20}")
    print(f"  Actual Pos   : {fn:<20} {tp:<20}")

    print("\n" + "-" * 70)
    print("  HEAD-TO-HEAD COMPARISON AGAINST BASELINES:")
    print("-" * 70)
    print(f"  {'Model':<30} {'Macro-F1':<12} {'Neg Recall':<14} {'Accuracy':<10}")
    print(f"  {'-'*29} {'-'*11} {'-'*13} {'-'*9}")
    print(f"  {'DeBERTa-v3-Omni (Full Train)':<30} {macro_f1:<12.4f} {neg_recall*100:<13.2f}% {acc*100:<9.2f}%  <-- CHAMPION")
    print(f"  {'DeBERTa-v3-Omni (50k Sub)':<30} {'0.9587':<12} {'93.54%':<14} {'97.81%':<10}")
    print(f"  {'DistilBERT (281k Baseline)':<30} {'0.9504':<12} {'91.38%':<14} {'97.23%':<10}")
    print(f"  {'Linear SVM (C=0.5)':<30} {'0.9253':<12} {'84.44%':<14} {'96.17%':<10}")
    print(f"  {'Naive Bayes (Baseline)':<30} {'0.8692':<12} {'69.42%':<14} {'93.71%':<10}")
    print("-" * 70)

    results_dict = {
        "model": "DeBERTa-v3-Omni Full Production",
        "checkpoint": str(MODEL_DIR / "best_model.pt"),
        "test_samples": len(test_texts),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "negative_recall": round(float(neg_recall), 4),
        "negative_precision": round(float(neg_prec), 4),
        "negative_f1": round(float(neg_f1), 4),
        "positive_recall": round(float(pos_recall), 4),
        "positive_precision": round(float(pos_prec), 4),
        "positive_f1": round(float(pos_f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "evaluation_minutes": round(float(eval_time), 2)
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n[Saved] Test benchmark results saved to {OUT_JSON}")
    return results_dict


if __name__ == "__main__":
    evaluate_test()
