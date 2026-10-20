import json
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
nb_path = ROOT / "notebooks" / "03b_distilbert_finetuning.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Guard cell 19
nb['cells'][19]['source'] = [
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    result_15 = train_manual_weight_run(\n",
    "        tag='distilbert_manual_weight_1p5',\n",
    "        negative_weight=1.5,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(result_15)\n",
    "else:\n",
    "    print('Skipping manual weight run 1.5x (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Guard cell 20
nb['cells'][20]['source'] = [
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    result_20 = train_manual_weight_run(\n",
    "        tag='distilbert_manual_weight_2x',\n",
    "        negative_weight=2.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(result_20)\n",
    "else:\n",
    "    print('Skipping manual weight run 2.0x (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Guard cell 21
nb['cells'][21]['source'] = [
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    result_30 = train_manual_weight_run(\n",
    "        tag='distilbert_manual_weight_3x',\n",
    "        negative_weight=3.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(result_30)\n",
    "else:\n",
    "    print('Skipping manual weight run 3.0x (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Guard cell 22
nb['cells'][22]['source'] = [
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    result_40 = train_manual_weight_run(\n",
    "        tag='distilbert_manual_weight_4x',\n",
    "        negative_weight=4.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(result_40)\n",
    "else:\n",
    "    print('Skipping manual weight run 4.0x (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Guard cell 23
nb['cells'][23]['source'] = [
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    result_50 = train_manual_weight_run(\n",
    "        tag='distilbert_manual_weight_5x',\n",
    "        negative_weight=5.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(result_50)\n",
    "else:\n",
    "    print('Skipping manual weight run 5.0x (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Guard cell 26
nb['cells'][26]['source'] = [
    "# Full 281k training with negative class weight\n",
    "FULL_WEIGHTED_TAG = 'distilbert_full281k_manual_4x'\n",
    "\n",
    "if RUN_MANUAL_WEIGHT_EXPERIMENTS:\n",
    "    full_weighted_result = train_manual_weight_run(\n",
    "        tag=FULL_WEIGHTED_TAG,\n",
    "        negative_weight=3.0,\n",
    "        positive_weight=1.0\n",
    "    )\n",
    "    display(full_weighted_result)\n",
    "else:\n",
    "    print('Skipping full weighted training run (RUN_MANUAL_WEIGHT_EXPERIMENTS=False)')\n"
]

# Update cell 29 checkpoint folder label commentary
src29 = nb['cells'][29]['source']
already_has_note = any("checkpoint-6250 corresponds to the 50k exploratory sweep" in line for line in src29)
if not already_has_note:
    for idx, line in enumerate(src29):
        if "FOUR_X_DIR = RUNS" in line:
            src29.insert(idx, "# Note: checkpoint-6250 corresponds to the 50k exploratory sweep run (3,125 steps/epoch x 2 epochs) assessing class weights prior to full scaling.\n")
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebooks/03b_distilbert_finetuning.ipynb!")
