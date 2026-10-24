"""
scripts/check_repo_size.py
==========================
Verify repository hygiene by scanning all tracked and untracked (non-ignored)
files via `git ls-files -o -c --exclude-standard` to guarantee no file exceeds 50 MB.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def check_sizes(threshold_mb: float = 50.0) -> int:
    try:
        res = subprocess.run(
            ["git", "ls-files", "-o", "-c", "--exclude-standard"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True
        )
    except Exception as e:
        print(f"Error executing git ls-files: {e}", file=sys.stderr)
        return 1

    files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    over_threshold = []

    for f_rel in files:
        f_path = ROOT / f_rel
        if f_path.is_file():
            sz_mb = f_path.stat().st_size / (1024 * 1024)
            if sz_mb > threshold_mb:
                over_threshold.append((f_rel, sz_mb))

    if over_threshold:
        print(f"FAILED: Found {len(over_threshold)} file(s) over {threshold_mb} MB:")
        for f, sz in over_threshold:
            print(f"  - {f}: {sz:.2f} MB")
        return 1

    print(f"PASSED: 0 files over {threshold_mb} MB in git ls-files -o -c --exclude-standard.")
    return 0

if __name__ == "__main__":
    sys.exit(check_sizes())
