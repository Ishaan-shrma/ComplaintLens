import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 05 - Phase 2: Custom DeBERTa-v3 Universal Sentiment Training\n",
            "---\n",
            "### Objectives & Architecture\n",
            "1. **Blended Replay Corpus:** 85% Amazon Fine Food (50k) + 15% Omni-Domain buffer (Yelp, IMDb, Flipkart).\n",
            "2. **Wild-Web Preprocessing:** Emoji-to-token, Hinglish/slang expansion, elongation compression, rating tokenization.\n",
            "3. **DeBERTa-v3 Architecture:** Attention-Masked Mean+Max Pooling & 5-Branch Multi-Sample Dropout.\n",
            "4. **Optimization:** Layer-Wise Learning Rate Decay (LLRD $\\alpha = 0.90$), Focal Loss ($\\gamma = 2.0$) + Label Smoothing ($\\epsilon = 0.05$).\n",
            "5. **Real-Time Visibility:** Live visual tqdm progress bar."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import torch\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "if str(ROOT) not in sys.path:\n",
            "    sys.path.append(str(ROOT))\n",
            "\n",
            "print(f'Project root: {ROOT}')\n",
            "print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'Device: {torch.cuda.get_device_name(0)}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Launch Phase 2 Training with live interactive tqdm progress bar\n",
            "from src.training.train_deberta import train\n",
            "\n",
            "train()\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/05_phase2_deberta_omni_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook created successfully!")
