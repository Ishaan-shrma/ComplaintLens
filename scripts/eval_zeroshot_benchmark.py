"""
scripts/eval_zeroshot_benchmark.py
==================================
Evaluates three pre-trained zero-shot transformer models on the canonical 72,846 Amazon test set:
1. DistilBERT-SST-2 (distilbert-base-uncased-finetuned-sst-2-english)
2. Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
3. NLP Town Multilingual BERT (nlptown/bert-base-multilingual-uncased-sentiment)

Saves results to reports/zeroshot_canonical_72k_evaluations.json
"""

import time
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

ROOT = Path("A:/DS-Review-AI")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

# Load canonical test set
with open(ROOT / "notebooks" / "artifacts" / "test_72846.pkl", "rb") as f:
    test_df = pickle.load(f)

texts = test_df["text"].tolist() if "text" in test_df else test_df["Text"].tolist()
y_true = np.load(ROOT / "notebooks" / "artifacts" / "test_labels.npy")
print(f"Loaded {len(texts):,} test reviews ({sum(y_true==0):,} neg / {sum(y_true==1):,} pos)")

def eval_model(model_id, name, mapping_fn, batch_size=128):
    print(f"\n--- Evaluating {name} ({model_id}) ---")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    mdl.eval()
    
    preds = []
    for i in range(0, len(texts), batch_size):
        b_texts = [str(t) for t in texts[i:i+batch_size]]
        enc = tok(b_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            b_preds = mapping_fn(probs)
            preds.extend(b_preds)
    
    preds = np.array(preds)
    dur = time.time() - t0
    
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    neg_prec = precision_score(y_true, preds, pos_label=0)
    neg_rec = recall_score(y_true, preds, pos_label=0)
    pos_prec = precision_score(y_true, preds, pos_label=1)
    pos_rec = recall_score(y_true, preds, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    print(f"Completed in {dur:.1f}s")
    print(f"  Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | NegRec: {neg_rec:.4f} | NegPrec: {neg_prec:.4f} | FP: {fp} | FN: {fn}")
    
    # Free memory
    del mdl, tok
    torch.cuda.empty_cache()
    
    return {
        "model_id": model_id,
        "display_name": name,
        "train_n": 0,
        "eval_set_id": "amazon_test",
        "eval_set_name": "Held-out test set (canonical)",
        "eval_n": len(y_true),
        "status": "zero-shot",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "negative_precision": round(float(neg_prec), 4),
        "negative_recall": round(float(neg_rec), 4),
        "positive_precision": round(float(pos_prec), 4),
        "positive_recall": round(float(pos_rec), 4),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tp": int(tp),
        "duration_seconds": round(dur, 1)
    }

results = []

# 1. DistilBERT SST-2
# Output: [NEGATIVE, POSITIVE]
def sst2_map(probs):
    return (probs[:, 1] >= probs[:, 0]).astype(int).tolist()

r_sst2 = eval_model("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT-SST-2 (Zero-Shot)", sst2_map)
r_sst2["label_mapping"] = "Binary argmax: index 0 (NEGATIVE) -> 0, index 1 (POSITIVE) -> 1"
results.append(r_sst2)

# 2. Twitter-RoBERTa
# Output: [negative, neutral, positive]
def roberta_map(probs):
    # Compare P(pos) vs P(neg)
    return (probs[:, 2] >= probs[:, 0]).astype(int).tolist()

r_rob = eval_model("cardiffnlp/twitter-roberta-base-sentiment-latest", "Twitter-RoBERTa (Zero-Shot)", roberta_map)
r_rob["label_mapping"] = "3-class: P(positive) >= P(negative) -> 1, else 0 (neutral non-binary mass marginalized)"
results.append(r_rob)

# 3. NLP Town BERT
# Output: [1 star, 2 stars, 3 stars, 4 stars, 5 stars]
def nlptown_map(probs):
    neg_mass = probs[:, 0] + probs[:, 1]
    pos_mass = probs[:, 3] + probs[:, 4]
    return (pos_mass >= neg_mass).astype(int).tolist()

r_nlp = eval_model("nlptown/bert-base-multilingual-uncased-sentiment", "NLP Town BERT (Zero-Shot)", nlptown_map)
r_nlp["label_mapping"] = "5-star rating: sum(4,5 stars) >= sum(1,2 stars) -> 1, else 0 (3-star neutral marginalized)"
results.append(r_nlp)

out_f = ROOT / "reports" / "zeroshot_canonical_72k_evaluations.json"
with open(out_f, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nAll zero-shot models evaluated on canonical 72,846 test set! Saved to: {out_f}")
