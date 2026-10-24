import pymupdf
import os

src_pdf = r"C:\Users\Ishaan\.gemini\antigravity\brain\1b2e669f-6a0d-43d0-b1b1-486e29fc8ecb\.user_uploaded\media_1792325987660_5e06120b.pdf"
sample_doc = pymupdf.open(src_pdf)

doc = pymupdf.open()
doc.insert_pdf(sample_doc, from_page=0, to_page=0)
page = doc[0]

# Redact the old metadata block and footer
redact_rect_meta = pymupdf.Rect(50, 440, 545, 635)
page.add_redact_annot(redact_rect_meta, fill=(1, 1, 1))

redact_rect_footer = pymupdf.Rect(50, 785, 545, 825)
page.add_redact_annot(redact_rect_footer, fill=(1, 1, 1))

page.apply_redactions()

pw = page.rect.width

# Insert Degree: "BACHELOR OF TECHNOLOGY"
deg_text = "BACHELOR OF TECHNOLOGY"
deg_w = pymupdf.get_text_length(deg_text, fontname="times-bold", fontsize=22)
page.insert_text(pymupdf.Point((pw - deg_w) / 2, 465), deg_text, fontname="times-bold", fontsize=22, color=(0, 0, 0))

sub_deg = "(ARTIFICIAL INTELLIGENCE & MACHINE LEARNING)"
sub_deg_w = pymupdf.get_text_length(sub_deg, fontname="times-bold", fontsize=11)
page.insert_text(pymupdf.Point((pw - sub_deg_w) / 2, 483), sub_deg, fontname="times-bold", fontsize=11, color=(0, 0, 0))

# Insert Metadata Lines (Bold label, normal value, centered)
meta_items = [
    ("SUBMITTED TO: ", "MR. NIKHIL BENIWAL (REPORTING MANAGER)"),
    ("SUBMITTED BY: ", "ISHAAN SHARMA"),
    ("COLLEGE: ", "DELHI TECHNICAL CAMPUS (DTC-GGSIPU)"),
    ("DURATION: ", "24th Aug 2026 to 26th Oct 2026"),
    ("MENTOR: ", "MR. DAKSH SHARMA"),
]

y_pos = 512
line_spacing = 22

for label, val in meta_items:
    label_w = pymupdf.get_text_length(label, fontname="times-bold", fontsize=12)
    val_w = pymupdf.get_text_length(val, fontname="times-roman", fontsize=12)
    total_w = label_w + val_w
    start_x = (pw - total_w) / 2
    
    page.insert_text(pymupdf.Point(start_x, y_pos), label, fontname="times-bold", fontsize=12, color=(0, 0, 0))
    page.insert_text(pymupdf.Point(start_x + label_w, y_pos), val, fontname="times-roman", fontsize=12, color=(0, 0, 0))
    y_pos += line_spacing

# Insert Footer: "Page 1 of 33"
footer_text = "Page 1 of 33"
footer_w = pymupdf.get_text_length(footer_text, fontname="helv", fontsize=11)
page.insert_text(pymupdf.Point((pw - footer_w) / 2, 800), footer_text, fontname="helv", fontsize=11, color=(0, 0, 0))

out_pdf = r"A:\DS-Review-AI\test_exact_cover.pdf"
doc.save(out_pdf)
print(f"Saved exact cover PDF to: {out_pdf}")

# Render to PNG to inspect
pix = page.get_pixmap(dpi=150)
out_png = r"A:\DS-Review-AI\reports\figures\report_screenshots\test_exact_cover.png"
pix.save(out_png)
print(f"Rendered exact cover PNG to: {out_png}")
