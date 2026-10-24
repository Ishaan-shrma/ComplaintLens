import fitz
from markdown_pdf import Section, MarkdownPdf

md1 = """# Test Markdown Image

![Workspace 01](reports/figures/report_screenshots/01_corpus_cleaning_analytics.png)
"""

md2 = """# Test HTML 100%

<img src="reports/figures/report_screenshots/01_corpus_cleaning_analytics.png" width="100%" />
"""

md3 = """# Test HTML 750

<img src="reports/figures/report_screenshots/01_corpus_cleaning_analytics.png" width="750" />
"""

pdf = MarkdownPdf()
pdf.add_section(Section(md1, root=r"A:\DS-Review-AI"))
pdf.add_section(Section(md2, root=r"A:\DS-Review-AI"))
pdf.add_section(Section(md3, root=r"A:\DS-Review-AI"))
pdf.save("test_img_render.pdf")

doc = fitz.open("test_img_render.pdf")
print("Total Pages:", len(doc))
for i, page in enumerate(doc):
    il = page.get_images()
    print(f"Page {i+1}: images={len(il)}")
    for img in il:
        bbox = page.get_image_bbox(img)
        print(f"   Image bbox: {bbox}, width={bbox.width:.1f}, height={bbox.height:.1f}")
