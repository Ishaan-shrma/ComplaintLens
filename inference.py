"""
Amazon Fine Food Reviews — Standalone Inference Script

This script provides a command-line interface for the calibrated DistilBERT 
sentiment classification model. It demonstrates how to invoke the model 
programmatically outside of the Streamlit dashboard environment.

Expected Input:
    --text (str): The raw review text string to be classified. 
                  (Automatically truncated to 128 tokens internally).

Expected Output (JSON):
    {
        "label": "positive" | "negative",
        "confidence": float (calibrated probability),
        "model_used": str,
        "caveat": str (noting binary constraints)
    }

Usage Example:
    python inference.py --text "This coffee is absolutely fantastic!"
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the src module can be found when run from the command line
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.tools import predict_sentiment

def run_inference(text: str) -> None:
    """Runs the text through the active production model and prints formatted JSON."""
    try:
        # The tool handles loading the model, tokenization, and temperature scaling (T=1.654)
        result = predict_sentiment(text)
        
        # Output strictly formatted JSON for downstream API/Microservice consumption
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        error_payload = {"error": "Inference failed", "details": str(e)}
        print(json.dumps(error_payload, indent=4))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment inference on Amazon reviews.")
    parser.add_argument(
        "--text", 
        type=str, 
        required=True, 
        help="The review text to classify."
    )
    
    args = parser.parse_args()
    run_inference(args.text)