import os
import pymupdf
from PIL import Image

# Let's inspect the exact dimensions and aspect ratios of all 9 screenshots
screenshots = {
    "6.1": ("01_corpus_cleaning_analytics.png", "Workspace [01] Corpus Architecture, Cleaning & Split Partitions"),
    "6.2": ("03_model_benchmarks_leaderboard.png", "Workspace [03] Model Comparison Leaderboard (11 Architectures)"),
    "6.3": ("03_distilbert_optimization_studio.png", "Workspace [03] DistilBERT Optimization Studio (Scaling & Class-Weights)"),
    "6.4": ("02_deberta_champion_kpis.png", "Workspace [02] Champion DeBERTa-v3-Omni Hero KPIs & Error Reductions"),
    "6.5": ("04_live_inference_lab.png", "Workspace [04] Live Inference Lab, Clause Dissector & Decision Threshold"),
    "6.6": ("05_cross_domain_analytics.png", "Workspace [05] Cross-Domain Multi-Platform Analytics & Lexicon Distinctiveness"),
    "6.7": ("06_ai_review_analyst_rag.png", "Workspace [06] Context-Grounded AI Review Analyst RAG & Cited Evidence"),
    "6.8": ("07_error_taxonomy_diagnostics.png", "Workspace [07] Rule-Based Error Taxonomy & Contrastive Clause Drops"),
    "6.9": ("08_engineering_limitations_drawer.png", "Section 9 Drawer: Engineering Limitations & Scope Disclosures")
}

d = r"A:\DS-Review-AI\reports\figures\report_screenshots"
for mod, (fn, title) in screenshots.items():
    fp = os.path.join(d, fn)
    assert os.path.exists(fp), f"Missing {fp}"
    im = Image.open(fp)
    print(f"Module {mod}: {fn:38s} | size={im.size} | aspect={im.size[0]/im.size[1]:.2f}")
print("All 9 screenshots verified present and readable!")
