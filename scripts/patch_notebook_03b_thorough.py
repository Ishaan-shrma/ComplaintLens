import json
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
nb_path = ROOT / "notebooks" / "03b_distilbert_finetuning.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 1: Add RESULTS_JSON and results loading
src1 = nb['cells'][1]['source']
results_line_present = any("RESULTS_JSON" in l for l in src1)
if not results_line_present:
    src1.append("\n# Results manifest for 50k runs and final model\n")
    src1.append("RESULTS_JSON = ART / 'distilbert_results.json'\n")
    src1.append("results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []\n")
nb['cells'][1]['source'] = src1

# 2. Update Cell 26: Rename tag to distilbert_50k_manual_3x and use negative_weight=3.0
nb['cells'][26]['source'] = [
    "# Exploratory 50k training with 3x negative class weight\n",
    "WEIGHTED_50K_TAG = 'distilbert_50k_manual_3x'\n",
    "\n",
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    weighted_50k_result = train_manual_weight_run(\n",
    "        tag=WEIGHTED_50K_TAG,\n",
    "        negative_weight=3.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(weighted_50k_result)\n",
    "else:\n",
    "    print('Skipping 50k 3x manual weighted training run (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# 3. Update Cell 29: Rename FOUR_X_DIR to THREE_X_DIR pointing to distilbert_50k_manual_3x, update 4x->3x, and guard
cell29_code = """# ============================================================
# EVALUATION - DISTILBERT 3x NEGATIVE CLASS WEIGHT (50k RUN)
# ============================================================

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

if RUN_MANUAL_WEIGHT_EXPERIMENTS:
    # 1. Load the saved 3x manual model checkpoint from 50k sweep
    THREE_X_DIR = RUNS / "distilbert_50k_manual_3x" / "checkpoint-6250"
    print("Loading 3x model from:", THREE_X_DIR)

    tokenizer_3x = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_3x = AutoModelForSequenceClassification.from_pretrained(THREE_X_DIR)

    # 2. Prepare the untouched test set
    test_ds_3x = HFDataset.from_pandas(test.reset_index(drop=True)).map(
        lambda batch: tokenizer_3x(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        batch_size=2000,
        remove_columns=["text"]
    )
    print(f"\\nTest samples: {len(test_ds_3x):,}")

    # 3. Run inference
    trainer_3x = Trainer(
        model=model_3x,
        data_collator=DataCollatorWithPadding(tokenizer_3x),
        args=TrainingArguments(
            output_dir=str(RUNS / "eval_3x"),
            per_device_eval_batch_size=64,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
    )

    output_3x = trainer_3x.predict(test_ds_3x)
    logits_3x = np.asarray(output_3x.predictions, dtype=np.float64)
    y_test_3x = np.asarray(output_3x.label_ids)
    y_pred_3x = logits_3x.argmax(axis=1)

    # 4. Probabilities
    shifted_logits = logits_3x - logits_3x.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs_3x = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    positive_probs_3x = probs_3x[:, 1]

    # 5. Metrics
    accuracy_3x = accuracy_score(y_test_3x, y_pred_3x)
    macro_f1_3x = f1_score(y_test_3x, y_pred_3x, average="macro")
    roc_auc_3x = roc_auc_score(y_test_3x, positive_probs_3x)
    report_3x = classification_report(y_test_3x, y_pred_3x, target_names=["negative", "positive"], digits=4)
    tn_3x, fp_3x, fn_3x, tp_3x = (int(v) for v in confusion_matrix(y_test_3x, y_pred_3x).ravel())

    # 6. Display results
    print("\\n" + "=" * 60)
    print("DISTILBERT 3x NEGATIVE CLASS WEIGHT (50k RUN) TEST EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy_3x:.4f} | Macro-F1: {macro_f1_3x:.4f} | ROC-AUC: {roc_auc_3x:.4f}")
    print(f"TN={tn_3x:,}  FP={fp_3x:,} | FN={fn_3x:,}  TP={tp_3x:,}")
    print("=" * 60)
else:
    print("Skipping 3x manual weight test evaluation (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)")
"""
nb['cells'][29]['source'] = [line + "\n" for line in cell29_code.split("\n")]

# 4. Check for any empty cells and remove them
non_empty_cells = []
for c in nb['cells']:
    src = "".join(c.get("source", [])).strip()
    if src:
        non_empty_cells.append(c)
print(f"Retained {len(non_empty_cells)} non-empty cells out of {len(nb['cells'])} total cells.")
nb['cells'] = non_empty_cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully patched notebooks/03b_distilbert_finetuning.ipynb!")
