import json
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
nb_path = ROOT / "notebooks" / "03b_distilbert_finetuning.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in 03b: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    first_lines = "".join(c['source'][:3]).strip().replace("\n", " ")
    print(f"Cell {i:02d} [{c['cell_type']}]: {first_lines[:90]}")
