#!/usr/bin/env python3
"""
scripts/download_assets.py
==========================
Download and verify heavy model weights, FAISS vector index, and full datasets
for the Review Intelligence Prototype from Hugging Face Hub or GitHub Releases.

Usage:
    python scripts/download_assets.py [--token <HF_TOKEN>] [--repo <HF_REPO_ID>] [--verify-only]

Environment Variables:
    HF_TOKEN     Hugging Face access token (required for private repositories)
    HF_REPO_ID   Hugging Face repository ID (default: Ishaan/review-intelligence-assets)
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Expected SHA-256 checksums for full binary integrity verification
EXPECTED_SHA256 = {
    "models/deberta_omni_v1/best_model.pt": "875c29404178b3612088529f9f57f82da9708750dad28f30651bfdddd311f047",
    "notebooks/artifacts/distilbert_full281k/model.safetensors": "cfa50196a63b7b2e2fb44c1341be678e1a2f26b11793148eb76ad62b72c4fbed",
    "models/faiss_index_with_ids.bin": "8d7858c4ce3745a3bf1a817390c333504803522eb6c6a69650cf7509e09fc6c3",
    "models/embeddings.npy": "308ac42081edc7f1b8cf5206c68e229c939070cce2348b838f195d3eaaf19e22",
    "data/processed/reviews_cleaned.csv": "42ec37b3619b702c8b9beb719d8128c7baa1de0802ee9ba88fec1e88cf3373e6",
}

DEFAULT_HF_REPO = os.getenv("HF_REPO_ID", "Ishaan/review-intelligence-assets")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_file(rel_path: str) -> bool:
    target = PROJECT_ROOT / rel_path
    if not target.exists():
        print(f"[MISSING] {rel_path}")
        return False
    exp = EXPECTED_SHA256.get(rel_path)
    if not exp:
        print(f"[OK] {rel_path} exists (no SHA-256 recorded)")
        return True
    act = compute_sha256(target)
    if act.lower() == exp.lower():
        print(f"[VERIFIED] {rel_path} (SHA-256 matched: {act[:12]}...)")
        return True
    else:
        print(f"[CORRUPT] {rel_path} (Expected {exp[:12]}, got {act[:12]})")
        return False


def download_from_hf(repo_id: str, rel_path: str, token: str = None) -> bool:
    target = PROJECT_ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"\nDownloading: {rel_path} from {url}")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(target, "wb") as out_f:
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Progress: {mb:.1f} MB / {total / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
    except urllib.error.HTTPError as e:
        if e.code == 401 or e.code == 403:
            print(f"\n[Error 401/403] Authorization failed for repository '{repo_id}'.")
            print("  A valid Hugging Face token is required for private repositories.")
            print("  Please provide a token via --token <HF_TOKEN> or set the HF_TOKEN environment variable.")
        elif e.code == 404:
            print(f"\n[Error 404] Asset '{rel_path}' not found in repository '{repo_id}'.")
        else:
            print(f"\n[HTTP Error {e.code}] Failed to download '{rel_path}': {e.reason}")
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        return False
    except Exception as e:
        print(f"\n[Error] Download failed for '{rel_path}': {e}")
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        return False

    return verify_file(rel_path)


def main():
    parser = argparse.ArgumentParser(description="Download heavy model weights and assets.")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face API token (for private repos)")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO, help=f"HF Repository ID (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing local assets without downloading")
    parser.add_argument("--skip-data", action="store_true", help="Skip downloading full reviews_cleaned.csv")
    args = parser.parse_args()

    print("=" * 70)
    print(" Review Intelligence Prototype - Asset Downloader & Integrity Verifier")
    print("=" * 70)

    if args.verify_only:
        print("\nVerifying local assets checksums:")
        all_ok = True
        for p in EXPECTED_SHA256:
            if not verify_file(p):
                all_ok = False
        if all_ok:
            print("\n[SUCCESS] All expected assets are present with valid SHA-256 hashes.")
            sys.exit(0)
        else:
            print("\n[NOTICE] One or more assets are missing or modified.")
            sys.exit(1)

    # Check token requirement
    token = args.token
    if not token:
        print("\n[Notice] No HF_TOKEN supplied.")
        print("  If the asset repository is private, downloads will return HTTP 401/403.")
        print("  Set HF_TOKEN in your environment or pass --token <HF_TOKEN>.\n")

    targets = list(EXPECTED_SHA256.keys())
    if args.skip_data:
        targets = [t for t in targets if not t.endswith(".csv")]

    failed = []
    for rel_path in targets:
        target = PROJECT_ROOT / rel_path
        if target.exists() and verify_file(rel_path):
            print(f"Skipping {rel_path} (already verified)")
            continue
        ok = download_from_hf(args.repo, rel_path, token=token)
        if not ok:
            failed.append(rel_path)

    if failed:
        print("\n" + "=" * 70)
        print(f"[FAIL] {len(failed)} asset(s) could not be downloaded or verified:")
        for f in failed:
            print(f"  - {f}")
        print("\nNote: The prototype runs in 'replay' mode when weights are missing,")
        print("serving exact pre-recorded benchmark predictions from public/data/.")
        print("=" * 70)
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("[SUCCESS] All model assets downloaded and verified successfully!")
        print("The prototype can now run in full 'live' inference mode.")
        print("=" * 70)


if __name__ == "__main__":
    main()
