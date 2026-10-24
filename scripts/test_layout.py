import re
import pymupdf
from markdown_pdf import Section, MarkdownPdf

with open(r"A:\DS-Review-AI\DS_GROUP_INTERNSHIP_FINAL_REPORT.md", "r", encoding="utf-8") as f:
    text = f.read()

# Strip broken anchors and internal links
clean_text = re.sub(r"\[([^\]]+)\]\(#[^\)]+\)", r"\1", text)
# Strip page breaks
clean_text = re.sub(r"<!--\s*PAGE\s*BREAK\s*-->", "", clean_text)
clean_text = re.sub(r"<div[^>]*page-break[^>]*></div>", "", clean_text)

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(clean_text, root=r"A:\DS-Review-AI"))
pdf.save(r"A:\DS-Review-AI\test_natural.pdf")

doc = pymupdf.open(r"A:\DS-Review-AI\test_natural.pdf")
print("Total Pages with natural flow:", len(doc))
for i, page in enumerate(doc):
    lines = [l.strip() for l in page.get_text().split("\n") if l.strip()]
    first = lines[0] if lines else "EMPTY"
    last = lines[-1] if lines else "EMPTY"
    print(f"Page {i+1:2d}: {len(lines):2d} lines | {first[:35]} ... {last[:35]}")
