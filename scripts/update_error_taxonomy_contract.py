"""
scripts/update_error_taxonomy_contract.py
=========================================
Updates public/data/error_taxonomy.json to establish DeBERTa-v3-Omni
as the primary model, with DistilBERT as the comparison baseline.
- Uses REAL dataset Id values from data/processed/reviews_cleaned.csv
  for example review IDs (e.g. #265, #594, #5732).
- Includes overall reduction context in comparative notes:
  "Overall: 1,901 to 1,382 (-27.3%); this category: -26.8% (-366 errors)."
- Explicit hover/tooltip definitions for overlap:
  fixed: "DistilBERT wrong, DeBERTa correct"
  new: "DeBERTa wrong, DistilBERT correct"
  shared: "Both models wrong"
- Dual-model error counts and base rates.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path("A:/DS-Review-AI")
df_deb = pd.read_csv(ROOT / "reports" / "taxonomy_deberta.csv")
df_dis = pd.read_csv(ROOT / "reports" / "taxonomy_distilbert.csv")
df_all = pd.read_csv(ROOT / "reports" / "taxonomy_full_audit_deberta.csv")

# Load real dataset Id mapping from reviews_cleaned.csv
df_clean = pd.read_csv(ROOT / "data" / "processed" / "reviews_cleaned.csv", usecols=["Id", "full_text"])
text_to_id = dict(zip(df_clean["full_text"], df_clean["Id"]))

df_deb["dataset_id"] = df_deb["text"].map(text_to_id)
df_dis["dataset_id"] = df_dis["text"].map(text_to_id)
df_all["dataset_id"] = df_all["text"].map(text_to_id)

# Save updated CSVs with real dataset_id
df_deb.to_csv(ROOT / "reports" / "taxonomy_deberta.csv", index=False)
df_dis.to_csv(ROOT / "reports" / "taxonomy_distilbert.csv", index=False)
df_all.to_csv(ROOT / "reports" / "taxonomy_full_audit_deberta.csv", index=False)

cat_order = [
    "mixed_clauses",
    "indirect_negation",
    "packaging_defect",
    "sequence_truncation",
    "rating_inversion",
    "domain_lexicon",
    "sarcasm"
]

cat_labels = {
    "mixed_clauses": "Mixed / Contrastive Clauses ('good, but...')",
    "indirect_negation": "Indirect & Nuanced Sentiment (Residual)",
    "packaging_defect": "Subtle Packaging/Transit Complaints",
    "sequence_truncation": "Sequence Truncation Beyond 128 Tokens",
    "rating_inversion": "Rating-Text Inversion (Label Noise)",
    "domain_lexicon": "Domain-Specific Food Idioms & Jargon",
    "sarcasm": "Sarcasm & Ironic Polarity"
}

cat_descriptions = {
    "mixed_clauses": "Reviews containing contrasting conjunctions (but, however, although, though, yet, etc.). The rule is broad and also applies to many correctly classified reviews.",
    "indirect_negation": "Subtle dissatisfaction expressed through nuance, qualified statements, or absence of overt negation keywords.",
    "packaging_defect": "Complaints focused on outer shipping boxes, broken inner seals, crushed packaging, or logistics transit damage.",
    "sequence_truncation": "Long customer reviews spanning 128 or more words where critical sentiment signals may lie beyond the token truncation horizon.",
    "rating_inversion": "Reviews where positive praise text is attached to 1-star ratings or vice versa, indicative of user rating error or label noise.",
    "domain_lexicon": "Domain-specific food descriptors such as stale, rancid, cardboard, chemical, rubbery, or aftertaste.",
    "sarcasm": "Ironic polarity, sarcastic praise (e.g. yeah right, thanks for nothing), or expressive punctuation (!?, ?!)."
}

deb_n_err = len(df_deb)
dis_n_err = len(df_dis)
n_total_test = len(df_all)

assert deb_n_err == 1382, f"Expected DeBERTa errors 1382, got {deb_n_err}"
assert dis_n_err == 1901, f"Expected DistilBERT errors 1901, got {dis_n_err}"

overall_diff = dis_n_err - deb_n_err
overall_red_pct = round((overall_diff / dis_n_err) * 100.0, 1)

categories_deb = []
categories_dis = []

print("\nExtracting verified real dataset Ids per category:")
for cid in cat_order:
    tot_cat = len(df_all[df_all["category_id"] == cid])
    
    # DeBERTa
    deb_sub = df_deb[df_deb["category_id"] == cid]
    deb_cnt = len(deb_sub)
    deb_share = round(deb_cnt / deb_n_err, 4)
    deb_pct = round(deb_cnt / deb_n_err * 100.0, 2)
    deb_rate = round(deb_cnt / tot_cat, 4)
    deb_rate_pct = round(deb_cnt / tot_cat * 100.0, 2)
    
    # Real dataset IDs
    deb_examples = [str(int(r)) for r in deb_sub["dataset_id"].dropna().iloc[:3].tolist()]
    print(f"  {cid:<22}: {deb_examples}")
    
    # DistilBERT
    dis_sub = df_dis[df_dis["category_id"] == cid]
    dis_cnt = len(dis_sub)
    dis_share = round(dis_cnt / dis_n_err, 4)
    dis_pct = round(dis_cnt / dis_n_err * 100.0, 2)
    dis_rate = round(dis_cnt / tot_cat, 4)
    dis_rate_pct = round(dis_cnt / tot_cat * 100.0, 2)
    dis_examples = [str(int(r)) for r in dis_sub["dataset_id"].dropna().iloc[:3].tolist()]

    # Generated Comparative Error Reduction Note with overall context
    diff_cnt = dis_cnt - deb_cnt
    if diff_cnt > 0:
        cat_red_pct = round((diff_cnt / dis_cnt) * 100.0, 1)
        comp_note = (
            f"DeBERTa has {deb_cnt:,} errors against {dis_cnt:,} for DistilBERT in this category "
            f"(raw text, amazon_test, n = {n_total_test:,}). "
            f"Overall test errors fell from {dis_n_err:,} to {deb_n_err:,} (-{overall_red_pct}%); "
            f"this category: -{cat_red_pct}% (-{diff_cnt:,} errors). "
            f"Not an ablation; model differences are bundled."
        )
    elif diff_cnt == 0:
        comp_note = (
            f"DeBERTa has {deb_cnt:,} errors against {dis_cnt:,} for DistilBERT in this category "
            f"(raw text, amazon_test, n = {n_total_test:,}). "
            f"Overall test errors fell from {dis_n_err:,} to {deb_n_err:,} (-{overall_red_pct}%); "
            f"this category: 0.0% (identical error count). "
            f"Not an ablation; model differences are bundled."
        )
    else:
        comp_note = (
            f"DeBERTa has {deb_cnt:,} errors against {dis_cnt:,} for DistilBERT in this category "
            f"(raw text, amazon_test, n = {n_total_test:,}). "
            f"Overall test errors fell from {dis_n_err:,} to {deb_n_err:,} (-{overall_red_pct}%); "
            f"this category does not support an error reduction (+{abs(diff_cnt):,} errors). "
            f"Not an ablation; model differences are bundled."
        )

    categories_deb.append({
        "id": cid,
        "label": cat_labels[cid],
        "description": cat_descriptions[cid],
        "count": deb_cnt,
        "count_tagged": deb_cnt,
        "count_reviewed": deb_cnt,
        "share": deb_share,
        "percentage": deb_pct,
        "total_reviews": tot_cat,
        "error_rate": deb_rate,
        "error_rate_pct": deb_rate_pct,
        "distilbert_count": dis_cnt,
        "distilbert_share": dis_share,
        "distilbert_percentage": dis_pct,
        "distilbert_error_rate_pct": dis_rate_pct,
        "error_reduction_count": diff_cnt,
        "comparative_note": comp_note,
        "example_ids": deb_examples
    })

    categories_dis.append({
        "id": cid,
        "label": cat_labels[cid],
        "count": dis_cnt,
        "count_tagged": dis_cnt,
        "count_reviewed": dis_cnt,
        "share": dis_share,
        "percentage": dis_pct,
        "total_reviews": tot_cat,
        "error_rate": dis_rate,
        "error_rate_pct": dis_rate_pct,
        "example_ids": dis_examples
    })

both_wrong = int((df_all["distilbert_error"] & df_all["deberta_raw_error"]).sum())
distil_only = int((df_all["distilbert_error"] & ~df_all["deberta_raw_error"]).sum())
deberta_only = int((~df_all["distilbert_error"] & df_all["deberta_raw_error"]).sum())

taxonomy_data = {
    "meta": {
        "model_id": "deberta-v3-omni",
        "display_name": "DeBERTa-v3-Omni (#1 Champion)",
        "champion_definition": "Best on the held-out test set among the models compared here.",
        "eval_set_id": "amazon_test",
        "split": "test",
        "n": n_total_test,
        "eval_set": {
            "id": "amazon_test",
            "name": "Held-out test set (canonical)",
            "n": n_total_test,
            "n_negative": int((df_all["y_true"] == 0).sum()),
            "n_positive": int((df_all["y_true"] == 1).sum())
        },
        "inference_config": "raw text, argmax",
        "source_files": [
            "reports/taxonomy_deberta.csv",
            "reports/taxonomy_distilbert.csv",
            "reports/taxonomy_full_audit_deberta.csv",
            "reports/test_predictions_per_row.csv"
        ],
        "provenance": {
            "tagger_script": "scripts/compute_taxonomy_deberta.py",
            "tagger_version": "1.0.0",
            "priority_order": [
                "1. Rating-Proxy Inversion (Label Noise)",
                "2. Mixed / Contrastive Clauses ('good, but...')",
                "3. Subtle Packaging/Transit Complaints",
                "4. Sarcasm & Ironic Polarity",
                "5. Sequence Truncation Beyond 128 Tokens",
                "6. Domain-Specific Food Idioms & Jargon",
                "7. Indirect & Nuanced Sentiment (Residual)"
            ]
        }
    },
    "model_id": "deberta-v3-omni",
    "display_name": "DeBERTa-v3-Omni (#1 Champion)",
    "champion_definition": "Best on the held-out test set among the models compared here.",
    "eval_set_id": "amazon_test",
    "split": "test",
    "n": n_total_test,
    "inference_config": "raw text, argmax",
    "n_errors_total": deb_n_err,
    "n_tagged": deb_n_err,
    "n_reviewed": deb_n_err,
    "labeling_type": "rule-based categories",
    "disclosure": "Rules were written after inspecting DistilBERT test errors (adaptive reuse of the test set). No manual validation was performed. Counts are assigned after earlier rules claim matching reviews (priority order).",
    "chart_title": "Misclassification Distribution by Rule-Based Category",
    "chart_subtitle": "Dual-model comparison of discrete error counts. Indigo: DeBERTa-v3-Omni (n = 1,382 errors, primary); Grey: DistilBERT 281k (n = 1,901 errors, comparison).",
    "chart_primary_legend": "DeBERTa-v3-Omni (Primary, n = 1,382)",
    "chart_comparison_legend": "DistilBERT 281k (Comparison, n = 1,901)",
    "chart_footer_note": "Card chip n = 1,382 reflects primary DeBERTa error population. Grey bars reflect DistilBERT errors (n = 1,901).",
    "categories": categories_deb,
    "comparison": {
        "model_id": "distilbert-base-uncased",
        "display_name": "DistilBERT 281k (Comparison Model)",
        "eval_set_id": "amazon_test",
        "split": "test",
        "n": n_total_test,
        "inference_config": "raw text, argmax",
        "n_errors_total": dis_n_err,
        "n_tagged": dis_n_err,
        "n_reviewed": dis_n_err,
        "categories": categories_dis
    },
    "overlap": {
        "shared_errors": both_wrong,
        "fixed_errors": distil_only,
        "new_errors": deberta_only,
        "net_reduction": distil_only - deberta_only,
        "mcnemar_chi2": 199.20,
        "p_value": "< 0.0001",
        "fixed_definition": "DistilBERT wrong, DeBERTa correct",
        "new_definition": "DeBERTa wrong, DistilBERT correct",
        "shared_definition": "Both models wrong",
        "overall_distilbert_errors": dis_n_err,
        "overall_deberta_errors": deb_n_err,
        "overall_reduction_pct": overall_red_pct,
        "description": f"Of DeBERTa's 1,382 errors, {both_wrong} are shared with DistilBERT and {deberta_only} are new errors where DistilBERT was correct. DeBERTa correctly classifies {distil_only} reviews that DistilBERT failed on, for a net reduction of {distil_only - deberta_only} errors."
    }
}

out_file = ROOT / "public" / "data" / "error_taxonomy.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(taxonomy_data, f, indent=2)

print(f"\nSaved updated {out_file} successfully!")
