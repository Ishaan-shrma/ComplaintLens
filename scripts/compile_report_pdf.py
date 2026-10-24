import re
import os
from markdown_pdf import Section, MarkdownPdf
import pymupdf

md_path = r"A:\DS-Review-AI\DS_GROUP_INTERNSHIP_FINAL_REPORT.md"
pdf_path = r"A:\DS-Review-AI\DS_GROUP_INTERNSHIP_FINAL_REPORT.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split by the page break delimiter
chunks = re.split(r"<!--\s*PAGE\s*BREAK\s*-->", full_text)
print(f"Total sections delimited by PAGE BREAK: {len(chunks)}")

pdf = MarkdownPdf(toc_level=2)
for i, chunk in enumerate(chunks):
    chunk_clean = chunk.strip()
    if not chunk_clean:
        continue
    # Strip broken markdown link anchors for PDF generator
    chunk_clean = re.sub(r"\[([^\]]+)\]\(#[^\)]+\)", r"\1", chunk_clean)
    # Remove HTML div page breaks if any to prevent duplicate blank pages
    chunk_clean = re.sub(r"<div[^>]*page-break[^>]*></div>", "", chunk_clean)
    pdf.add_section(Section(chunk_clean, root=r"A:\DS-Review-AI"))

pdf.save(pdf_path)
print("Page-break aligned PDF generated successfully!")

doc = pymupdf.open(pdf_path)
print("Final PDF Page Count:", len(doc))
print("Final PDF Size:", os.path.getsize(pdf_path) / 1024, "KB")

# Sync to handover folder
handover_pdf = r"A:\DS-Review-AI-Handover-Full\DS_GROUP_INTERNSHIP_FINAL_REPORT.pdf"
handover_md = r"A:\DS-Review-AI-Handover-Full\DS_GROUP_INTERNSHIP_FINAL_REPORT.md"

import shutil
shutil.copy2(pdf_path, handover_pdf)
shutil.copy2(md_path, handover_md)
print("Synced both MD and PDF to Handover folder successfully!")
