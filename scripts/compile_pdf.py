"""
Compiler script: Renders ML_Systems_Technical_Defense_Manual.html to PDF using Chrome Headless.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("A:/DS-Review-AI")
HTML_FILE = ROOT / "ML_Systems_Technical_Defense_Manual.html"
PDF_FILE = ROOT / "ML_Systems_Technical_Defense_Manual.pdf"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def compile_pdf():
    if not HTML_FILE.exists():
        print(f"Error: {HTML_FILE} does not exist.")
        sys.exit(1)

    print(f"Compiling {HTML_FILE.name} -> {PDF_FILE.name} ...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_FILE.resolve()}",
        str(HTML_FILE.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Chrome error: {res.stderr}")
        sys.exit(1)

    if not PDF_FILE.exists():
        print(f"Error: {PDF_FILE} was not created.")
        sys.exit(1)

    size_kb = PDF_FILE.stat().st_size / 1024
    print(f"Success! Generated: {PDF_FILE}")
    print(f"File Size: {size_kb:.1f} KB")

    try:
        import fitz
        doc = fitz.open(str(PDF_FILE))
        print(f"Verified with PyMuPDF: {len(doc)} pages rendered.")
        doc.close()
    except Exception as e:
        print(f"Note on verification: {e}")

if __name__ == "__main__":
    compile_pdf()
