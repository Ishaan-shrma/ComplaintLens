"""
scripts/tune_deberta_hyperparams.py
====================================
Systematic Hyperparameter Tuning for DeBERTa-v3-Omni (Phase 2).

Methodology: One-Factor-At-A-Time (OFAT) design.
  - Start from the proven champion configuration (bf0752b2, Val F1=0.9604).
  - Vary exactly ONE parameter per trial while holding others fixed.
  - This isolates the marginal effect of each hyperparameter on model quality.
  - Clean, explainable, and scientifically rigorous for job portfolio evidence.

Speed Optimizations Applied:
  - TF32 math for Ampere GPU (RTX 3060 Ti) — ~25% faster matmul, zero accuracy loss
  - BFloat16 / FP16 modern AMP — hardware accelerated, fast and memory-efficient
  - BATCH_SIZE=12, GRAD_ACCUM=3 (effective batch 36) — high GPU utilization
  - Pre-tokenized TensorDataset reused across all trials (tokenize once, reuse 4x)
  - tqdm live progress bars for clear ETA & step metrics

Outputs:
  - Each trial logged to runs/experiments.db (RunLogger)
  - Full ranked summary saved to runs/tuning_results.json
  - Human-readable report generated via scripts/report_tuning.py
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, accuracy_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.deberta_omni import DeBERTaOmniClassifier
from src.models.focal_loss import FocalLossWithSmoothing
from src.training.train_deberta import (
    load_blended_data, build_tensor_dataset, get_llrd_param_groups,
    MODEL_NAME, MAX_LENGTH, SEED, LABEL2ID
)
from src.training.logger import RunLogger

# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 12   # Up from 8 — maximizes GPU memory bandwidth
GRAD_ACCUM = 3    # Effective batch = 12 × 3 = 36 (well matched to 32-36 baseline)
EPOCHS     = 2    # 2 epochs per trial — fast diagnostic ranking

# ─────────────────────────────────────────────────────────────────────────────
# OFAT Trial Design
# Each trial is the CHAMPION config with exactly ONE parameter changed.
# Champion (bf0752b2): gamma=2.0, eps=0.05, alpha=0.90, head_lr=5e-5
# ─────────────────────────────────────────────────────────────────────────────
TRIALS = [
    {
        "name":             "T1_Champion_Baseline",
        "description":      "Exact champion config (Val F1=0.9604 @ 3 ep reference)",
        "focal_gamma":      2.0,
        "label_smoothing":  0.05,
        "llrd_alpha":       0.90,
        "head_lr":          5e-5,
        "encoder_top_lr":   2.5e-5,
    },
    {
        "name":             "T2_Lower_FocalGamma",
        "description":      "Reduce focal loss gamma 2.0 -> 1.5 (penalize hard samples less)",
        "focal_gamma":      1.5,
        "label_smoothing":  0.05,
        "llrd_alpha":       0.90,
        "head_lr":          5e-5,
        "encoder_top_lr":   2.5e-5,
    },
    {
        "name":             "T3_Tighter_LabelSmoothing",
        "description":      "Reduce label smoothing 0.05 -> 0.02 (sharper class probability bounds)",
        "focal_gamma":      2.0,
        "label_smoothing":  0.02,
        "llrd_alpha":       0.90,
        "head_lr":          5e-5,
        "encoder_top_lr":   2.5e-5,
    },
    {
        "name":             "T4_Aggressive_LLRD",
        "description":      "Reduce LLRD alpha 0.90 -> 0.85 (lower encoder layers decay faster)",
        "focal_gamma":      2.0,
        "label_smoothing":  0.05,
        "llrd_alpha":       0.85,
        "head_lr":          5e-5,
        "encoder_top_lr":   2.5e-5,
    },
]


def enable_speed_optimizations(device):
    """Enable Ampere-specific GPU hardware acceleration."""
    if device.type != "cuda":
        return

    # TF32: ~25% faster matrix multiplications on Ampere (RTX 30xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  [Speed] TF32 tensor core math enabled")

    # cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    print("  [Speed] cuDNN benchmark mode enabled")


def validate(model, val_loader, criterion, device, amp_dtype):
    """Run full validation pass. Returns (macro_f1, neg_recall, accuracy, avg_loss)."""
    model.eval()
    val_preds, val_targets = [], []
    v_loss = 0.0

    with torch.no_grad():
        for b_in, b_att, b_lbl in val_loader:
            b_in  = b_in.to(device, non_blocking=True)
            b_att = b_att.to(device, non_blocking=True)
            b_lbl = b_lbl.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out  = model(b_in, b_att)
                loss = criterion(out["logits"], b_lbl)

            v_loss += loss.item()
            val_preds.extend(torch.argmax(out["logits"], dim=-1).cpu().numpy())
            val_targets.extend(b_lbl.cpu().numpy())

    macro_f1   = f1_score(val_targets, val_preds, average="macro")
    neg_recall = recall_score(val_targets, val_preds, labels=[0], average="macro")
    accuracy   = accuracy_score(val_targets, val_preds)
    avg_loss   = v_loss / len(val_loader)

    return macro_f1, neg_recall, accuracy, avg_loss


def run_trial(cfg, trial_num, total_trials, train_ds, val_ds, device, logger, amp_dtype):
    """Run one full OFAT trial. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"  TRIAL {trial_num}/{total_trials}: {cfg['name']}")
    print(f"  Hypothesis: {cfg['description']}")
    print(f"  Params    : gamma={cfg['focal_gamma']} | eps={cfg['label_smoothing']} | alpha={cfg['llrd_alpha']}")
    print(f"{'='*70}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=0)

    run_id = logger.start_run(
        model_name      = "deberta-v3-base",
        epochs          = EPOCHS,
        lr              = cfg["head_lr"],
        batch_size      = BATCH_SIZE * GRAD_ACCUM,
        focal_gamma     = cfg["focal_gamma"],
        label_smoothing = cfg["label_smoothing"],
        llrd_alpha      = cfg["llrd_alpha"],
        max_length      = MAX_LENGTH,
        fp16            = True,
        seed            = SEED,
        notes           = f"OFAT Tuning | {cfg['name']} | {cfg['description']}"
    )

    torch.manual_seed(SEED)
    model = DeBERTaOmniClassifier(model_name=MODEL_NAME, num_labels=2)
    model.float()  # FP32 master weights for numerical stability
    model.to(device)

    param_groups = get_llrd_param_groups(model, cfg["head_lr"], cfg["encoder_top_lr"], cfg["llrd_alpha"])
    optimizer    = torch.optim.AdamW(param_groups)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * 0.06)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion    = FocalLossWithSmoothing(gamma=cfg["focal_gamma"], label_smoothing=cfg["label_smoothing"])
    scaler       = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    start_time     = time.time()
    trial_best_f1  = 0.0
    trial_best_rec = 0.0
    trial_best_acc = 0.0
    epoch_log      = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step_cnt   = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f"Trial {trial_num} [Ep {ep}/{EPOCHS}]", unit="batch")

        for b_idx, (b_in, b_att, b_lbl) in enumerate(train_pbar):
            b_in  = b_in.to(device, non_blocking=True)
            b_att = b_att.to(device, non_blocking=True)
            b_lbl = b_lbl.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out  = model(b_in, b_att)
                loss = criterion(out["logits"], b_lbl) / GRAD_ACCUM

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (b_idx + 1) % GRAD_ACCUM == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * GRAD_ACCUM
            step_cnt   += 1

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(train_loader):
                cur_loss = total_loss / step_cnt
                train_pbar.set_postfix({"loss": f"{cur_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # Validation
        val_f1, val_rec, val_acc, val_loss = validate(model, val_loader, criterion, device, amp_dtype)

        elapsed = (time.time() - start_time) / 60.0
        print(f"  [Trial {trial_num} Ep {ep}] Val Loss: {val_loss:.4f} | Macro-F1: {val_f1:.4f} | "
              f"Neg Recall: {val_rec*100:.2f}% | Acc: {val_acc*100:.2f}% | Elapsed: {elapsed:.1f}m")

        logger.log_epoch(
            run_id, ep,
            train_loss=total_loss / step_cnt,
            val_loss=val_loss,
            val_macro_f1=val_f1,
            val_neg_recall=val_rec
        )

        epoch_log.append({
            "epoch": ep,
            "val_f1": round(val_f1, 4),
            "val_neg_recall": round(val_rec, 4),
            "val_accuracy": round(val_acc, 4),
            "val_loss": round(val_loss, 4)
        })

        if val_f1 > trial_best_f1:
            trial_best_f1  = val_f1
            trial_best_rec = val_rec
            trial_best_acc = val_acc

    t_elapsed = (time.time() - start_time) / 60.0
    logger.finish_run(
        run_id,
        val_macro_f1=trial_best_f1,
        val_neg_recall=trial_best_rec,
        val_accuracy=trial_best_acc,
        training_minutes=t_elapsed
    )

    # Clean up GPU memory before next trial
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "trial_num":        trial_num,
        "run_id":           run_id,
        "name":             cfg["name"],
        "description":      cfg["description"],
        "focal_gamma":      cfg["focal_gamma"],
        "label_smoothing":  cfg["label_smoothing"],
        "llrd_alpha":       cfg["llrd_alpha"],
        "head_lr":          cfg["head_lr"],
        "val_macro_f1":     round(trial_best_f1, 4),
        "val_neg_recall":   round(trial_best_rec, 4),
        "val_accuracy":     round(trial_best_acc, 4),
        "training_minutes": round(t_elapsed, 1),
        "epoch_log":        epoch_log,
    }


def print_leaderboard(results):
    """Print clean ranked results table."""
    ranked = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)

    print("\n" + "=" * 78)
    print("  HYPERPARAMETER TUNING — FINAL LEADERBOARD")
    print("  Methodology: One-Factor-At-A-Time (OFAT)")
    print("  Champion Baseline (bf0752b2): Val Macro-F1 = 0.9604  [3-epoch benchmark]")
    print("=" * 78)
    print(f"  {'Rank':<5} {'Trial Name':<28} {'Macro-F1':>9} {'Neg Recall':>11} {'Accuracy':>9} {'Mins':>6}")
    print(f"  {'─'*4} {'─'*27} {'─'*9} {'─'*11} {'─'*9} {'─'*6}")

    for rank, r in enumerate(ranked, 1):
        winner_tag = "  <-- WINNER" if rank == 1 else ""
        print(f"  {rank:<5} {r['name']:<28} {r['val_macro_f1']:>9.4f} "
              f"{r['val_neg_recall']*100:>10.2f}% {r['val_accuracy']*100:>8.2f}% "
              f"{r['training_minutes']:>6.1f}{winner_tag}")

    print("=" * 78)
    winner = ranked[0]
    print(f"\n  WINNER CONFIGURATION: {winner['name']}")
    print(f"    * Focal Loss gamma : {winner['focal_gamma']}")
    print(f"    * Label Smoothing  : {winner['label_smoothing']}")
    print(f"    * LLRD alpha       : {winner['llrd_alpha']}")
    print(f"    * Val Macro-F1     : {winner['val_macro_f1']:.4f}")
    print(f"    * Val Neg Recall   : {winner['val_neg_recall']*100:.2f}%")
    print("=" * 78)

    return ranked[0]


def run_tuning():
    print("=" * 70)
    print("  DS-REVIEW-AI : DEBERTA-V3 HYPERPARAMETER TUNING")
    print("  Strategy  : One-Factor-At-A-Time (OFAT)")
    print(f"  Trials    : {len(TRIALS)}")
    print(f"  Epochs    : {EPOCHS} per trial")
    print(f"  Batch Size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective batch")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nDevice : {device} ({gpu_name})")
    enable_speed_optimizations(device)

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Precision: {amp_dtype} AMP active")

    # Tokenize ONCE, reuse across all trials
    print("\n[Step 1/3] Loading and normalizing blended training data...")
    train_texts, train_labels, val_texts, val_labels = load_blended_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n[Step 2/3] Pre-tokenizing datasets (reused across all 4 trials)...")
    train_ds = build_tensor_dataset(train_texts, train_labels, tokenizer, MAX_LENGTH, desc="Train Split")
    val_ds   = build_tensor_dataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH, desc="Val Split")
    print(f"Tokenization complete: {len(train_ds):,} train | {len(val_ds):,} val samples")

    print("\n[Step 3/3] Commencing OFAT Trials...")
    logger  = RunLogger()
    results = []

    for idx, cfg in enumerate(TRIALS, 1):
        result = run_trial(cfg, idx, len(TRIALS), train_ds, val_ds, device, logger, amp_dtype)
        results.append(result)

        # Save intermediate results after each trial to prevent data loss
        out_path = ROOT / "runs" / "tuning_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"trials_completed": idx, "results": results}, f, indent=2)
        print(f"  [Checkpoint] Intermediate results saved to runs/tuning_results.json")

    # Leaderboard & final JSON
    winner = print_leaderboard(results)

    final_payload = {
        "methodology":       "One-Factor-At-A-Time (OFAT)",
        "champion_baseline": {"run_id": "bf0752b2", "val_macro_f1": 0.9604, "val_neg_recall": 0.9369},
        "trials_completed":  len(results),
        "winner":            winner,
        "all_results":       sorted(results, key=lambda r: r["val_macro_f1"], reverse=True),
    }
    with open(ROOT / "runs" / "tuning_results.json", "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n[Completed] Full tuning results saved to runs/tuning_results.json")
    print("[Completed] Run 'python scripts/report_tuning.py' to generate Markdown report.")


if __name__ == "__main__":
    run_tuning()
