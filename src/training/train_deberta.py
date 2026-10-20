"""
src/training/train_deberta.py
=============================
Phase 2 Custom DeBERTa-v3 Training Pipeline:
  - 85% Amazon Fine Food + 15% Omni-Corpus blend
  - Layer-Wise Learning Rate Decay (LLRD alpha=0.90)
  - Focal Loss (gamma=2.0) + Label Smoothing (eps=0.05)
  - Multi-Sample Dropout Head (5 branches)
  - Integrated RunLogger (SQLite tracking)
  - Periodic heartbeat logs for background monitoring
"""

import os
import sys
import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.deberta_omni import DeBERTaOmniClassifier
from src.models.focal_loss import FocalLossWithSmoothing
from src.training.logger import RunLogger

# Hyperparameters
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 3  # Effective batch size = 36 (12 * 3)
EPOCHS = 2            # 2 full epochs over 320k reviews (281k Amazon + 42k Omni)
HEAD_LR = 5e-5
ENCODER_LR = 2.5e-5
LLRD_ALPHA = 0.90
MAX_LENGTH = 128
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
SEED = 42
USE_FULL_DATASET = True  # True: 281,271 full reviews; False: 50,000 sub-train

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "deberta_omni_v1"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT_LOG = PROJECT_ROOT / "runs" / "training_heartbeat.log"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def log_heartbeat(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(HEARTBEAT_LOG, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def build_tensor_dataset(texts, labels, tokenizer, max_length, desc="Tokenizing"):
    log_heartbeat(f"{desc}: Processing {len(texts):,} reviews in fast multi-threaded batch...")
    # Fast vectorized batch tokenization
    encodings = tokenizer(
        [str(t) for t in texts],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    from torch.utils.data import TensorDataset
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels_tensor)


def get_llrd_param_groups(model: DeBERTaOmniClassifier, head_lr: float, encoder_top_lr: float, alpha: float):
    layer_groups, num_layers = model.get_encoder_params_by_layer()
    param_groups = []

    # Head parameters (highest learning rate)
    param_groups.append({
        "params": model.get_head_params(),
        "lr": head_lr,
        "weight_decay": 0.01
    })

    # Encoder layers decaying downward
    for layer_idx, params in layer_groups:
        if layer_idx == -1:  # Embeddings
            lr = encoder_top_lr * (alpha ** (num_layers + 1))
        else:
            lr = encoder_top_lr * (alpha ** (num_layers - 1 - layer_idx))
        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": 0.01
        })

    return param_groups


def load_blended_data():
    log_heartbeat("Loading training and validation partitions...")
    # 1. Amazon Fine Food reviews
    if USE_FULL_DATASET:
        train_pkl = PROJECT_ROOT / "notebooks" / "artifacts" / "full_train_281271.pkl"
        log_heartbeat("Loading FULL 281,271 canonical Amazon reviews...")
    else:
        train_pkl = PROJECT_ROOT / "notebooks" / "artifacts" / "tr_sub_50k.pkl"
        log_heartbeat("Loading 50,000 sub-sample Amazon reviews...")
    val_pkl = PROJECT_ROOT / "notebooks" / "artifacts" / "val_sub_9997.pkl"

    with open(train_pkl, "rb") as f:
        amz_train = pickle.load(f)
    with open(val_pkl, "rb") as f:
        amz_val = pickle.load(f)

    # 2. Omni Corpus replay buffer
    omni_csv = PROJECT_ROOT / "data" / "processed" / "omni_corpus.csv"
    if omni_csv.exists():
        omni_df = pd.read_csv(omni_csv)
    else:
        omni_df = pd.DataFrame()

    # Normalize data structures: historical splits use lowercase 'text' and 'label' (0=negative, 1=positive)
    train_texts = amz_train["text"].tolist() if "text" in amz_train else amz_train["Text"].tolist()
    train_labels = amz_train["label"].astype(int).tolist()

    val_texts = amz_val["text"].tolist() if "text" in amz_val else amz_val["Text"].tolist()
    val_labels = amz_val["label"].astype(int).tolist()

    if not omni_df.empty:
        # Binary sentiment alignment (0=negative, 1=positive); drop neutral or map positive
        omni_filtered = omni_df[omni_df["sentiment_label"].isin(["negative", "positive"])].copy()
        n_omni = int(len(train_texts) * 0.15)
        omni_sample = omni_filtered.sample(min(n_omni, len(omni_filtered)), random_state=SEED)
        omni_texts = omni_sample["text"].tolist()
        omni_labels = omni_sample["sentiment_label"].map({"negative": 0, "positive": 1}).astype(int).tolist()

        train_texts.extend(omni_texts)
        train_labels.extend(omni_labels)
        log_heartbeat(f"Blended dataset: {len(train_texts):,} train ({len(omni_texts):,} omni replay) | {len(val_texts):,} val")
    else:
        log_heartbeat(f"Training dataset: {len(train_texts):,} train | {len(val_texts):,} val")

    # Rigorous wild-web preprocessing across all training & validation samples
    log_heartbeat("Applying Wild-Web Text Normalizer (emojis, Hinglish, slang, punctuation, rating tokens)...")
    from src.data.normalizer import batch_normalize
    train_texts = batch_normalize(train_texts)
    val_texts = batch_normalize(val_texts)
    log_heartbeat("Preprocessing and text normalization complete.")

    return train_texts, train_labels, val_texts, val_labels


def train():
    log_heartbeat("=" * 60)
    log_heartbeat("Starting Phase 2 DeBERTa-v3-Omni Training Routine")
    log_heartbeat("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_heartbeat(f"Active compute hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log_heartbeat("[Speed] TF32 Tensor Cores and cuDNN benchmark mode enabled.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = RunLogger()
    run_id = logger.start_run(
        model_name="deberta-v3-base",
        epochs=EPOCHS,
        lr=HEAD_LR,
        batch_size=BATCH_SIZE * GRAD_ACCUM_STEPS,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        llrd_alpha=LLRD_ALPHA,
        max_length=MAX_LENGTH,
        fp16=True,
        seed=SEED,
        notes="DeBERTa-v3 Mean+Max Pooling + Multi-Sample Dropout + Focal Loss + LLRD"
    )
    log_heartbeat(f"Run registered with ID: {run_id}")

    train_texts, train_labels, val_texts, val_labels = load_blended_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = build_tensor_dataset(train_texts, train_labels, tokenizer, MAX_LENGTH, desc="Training Set")
    val_dataset = build_tensor_dataset(val_texts, val_labels, tokenizer, MAX_LENGTH, desc="Validation Set")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

    model = DeBERTaOmniClassifier(model_name=MODEL_NAME, num_labels=2)
    model.float()  # Ensure float32 master weights so AMP GradScaler can unscale properly
    model.to(device)

    param_groups = get_llrd_param_groups(model, HEAD_LR, ENCODER_LR, LLRD_ALPHA)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = FocalLossWithSmoothing(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    # Use modern torch.amp with bfloat16 or standard fp16 handling suitable for DeBERTa embeddings
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        epoch_start = time.time()

        from tqdm.auto import tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs["logits"], labels)
                loss = loss / GRAD_ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
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

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            step_count += 1
            train_pbar.set_postfix({"loss": f"{(total_loss / step_count):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if (batch_idx + 1) % 500 == 0:
                avg_l = total_loss / step_count
                elapsed_min = (time.time() - start_time) / 60.0
                log_heartbeat(f"Epoch {epoch} | Step {batch_idx+1}/{len(train_loader)} | Loss: {avg_l:.4f} | Total Time: {elapsed_min:.1f}m")

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", unit="batch")

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_pbar:
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids, attention_mask)
                    v_loss = criterion(outputs["logits"], labels)

                val_loss += v_loss.item()
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        val_acc = accuracy_score(val_targets, val_preds)
        neg_recall = recall_score(val_targets, val_preds, labels=[0], average="macro")
        avg_val_loss = val_loss / len(val_loader)

        log_heartbeat(f"Epoch {epoch} Complete -> Val Loss: {avg_val_loss:.4f} | Macro-F1: {val_macro_f1:.4f} | Acc: {val_acc:.4f} | Neg Recall: {neg_recall:.4f}")

        logger.log_epoch(
            run_id=run_id,
            epoch=epoch,
            train_loss=total_loss / step_count,
            val_loss=avg_val_loss,
            val_macro_f1=val_macro_f1,
            val_neg_recall=neg_recall
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            log_heartbeat(f"[*] New checkpoint saved with Macro-F1: {best_val_f1:.4f}")

    training_time_min = (time.time() - start_time) / 60.0
    logger.finish_run(
        run_id=run_id,
        val_macro_f1=best_val_f1,
        best_epoch=epoch,
        training_minutes=training_time_min,
        checkpoint_path=str(CHECKPOINT_DIR)
    )
    log_heartbeat(f"Training successfully completed in {training_time_min:.1f} minutes.")


if __name__ == "__main__":
    train()
