import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------
# 1. Fix project-relative paths
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Safe imports from src
from src.agent.agent import answer as run_agent
from src.agent.tools import (
    get_review_kpi,
    predict_sentiment,
    predict_deberta_omni,
    predict_head_to_head,
    search_reviews
)
from src.data.normalizer import normalize
from src.models.dissector import dissect_review

st.set_page_config(page_title="Universal Review Intelligence — DS Group", layout="wide", page_icon="📊")

st.title("📊 Universal Review Intelligence Platform")
st.caption("Dual-Engine Transformer Sentiment Modeling (DistilBERT vs. DeBERTa-v3-Omni) & Wild-Web Review Intelligence")

# -------------------------------------------------------------
# Session State & Callbacks
# -------------------------------------------------------------
if "predict_text_distil" not in st.session_state:
    st.session_state["predict_text_distil"] = "The packaging arrived damaged and completely stale."
if "predict_text_deberta" not in st.session_state:
    st.session_state["predict_text_deberta"] = "The earphone sound quality is 🔥🔥 but wire build is totally bakwas 🤮"
if "arena_text" not in st.session_state:
    st.session_state["arena_text"] = "Product quality is awesome 🔥 but delivery was 3 days late and customer support was useless 💀"
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

def set_distil_text(text):
    st.session_state["predict_text_distil"] = text

def set_deberta_text(text):
    st.session_state["predict_text_deberta"] = text

def set_arena_text(text):
    st.session_state["arena_text"] = text

# -------------------------------------------------------------
# Global KPI Header with Multi-Platform Domain Scope
# -------------------------------------------------------------
st.markdown("### 🌐 Enterprise Intelligence Scope")
scope_choice = st.radio(
    "Select Review Corpus / Domain Scope:",
    [
        "🌐 Universal Multi-Platform (453,878 Reviews)",
        "🛒 Amazon Fine Food (393,890 Reviews)",
        "🛍️ Flipkart E-Commerce (15,000 Reviews)",
        "🍽️ Yelp Restaurant Dining (25,000 Reviews)",
        "🎬 IMDb Entertainment (19,988 Reviews)"
    ],
    horizontal=True,
    help="Toggle between full multi-platform intelligence and specific e-commerce/service domains."
)

# Dynamic KPI calculations based on selected scope
if "Universal" in scope_choice:
    tot_val = 453878
    pos_val = 74.0
    neg_val = 17.8
    score_metric_label = "Model Neg Recall"
    score_metric_val = "93.91%"
    aux_metric_label = "Platforms Monitored"
    aux_metric_val = "4 Sources"
elif "Amazon" in scope_choice:
    tot_val = 393890
    pos_val = 77.9
    neg_val = 14.5
    score_metric_label = "Average Rating"
    score_metric_val = "4.18 / 5"
    aux_metric_label = "Helpfulness Ratio"
    aux_metric_val = "77.9%"
elif "Flipkart" in scope_choice:
    tot_val = 15000
    pos_val = 59.9
    neg_val = 25.0
    score_metric_label = "Category"
    score_metric_val = "Electronics & Retail"
    aux_metric_label = "Neutral Share"
    aux_metric_val = "15.1%"
elif "Yelp" in scope_choice:
    tot_val = 25000
    pos_val = 39.7
    neg_val = 40.1
    score_metric_label = "Category"
    score_metric_val = "Restaurants & Dining"
    aux_metric_label = "Neutral Share"
    aux_metric_val = "20.2%"
else: # IMDb
    tot_val = 19988
    pos_val = 49.9
    neg_val = 50.1
    score_metric_label = "Category"
    score_metric_val = "Cinema & Streaming"
    aux_metric_label = "Classification Split"
    aux_metric_val = "Binary Polar"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Reviews Ingested", f"{tot_val:,}")
c2.metric("Positive Sentiment", f"{pos_val:.1f}%")
c3.metric("Negative Grievance Rate", f"{neg_val:.1f}%", delta_color="inverse")
c4.metric(score_metric_label, score_metric_val)
c5.metric(aux_metric_label, aux_metric_val)

st.divider()

# -------------------------------------------------------------
# Chart Data & Corpus Loader
# -------------------------------------------------------------
@st.cache_data
def load_sample_corpus():
    """Dynamically loads review sample with word count for interactive slider EDA."""
    try:
        csv_p = PROJECT_ROOT / "data" / "processed" / "embedding_sample_metadata.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p, usecols=["Id", "Score", "sentiment_label", "Summary", "Text"])
            df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
            df["sentiment_label"] = df["sentiment_label"].astype(str).str.lower().str.strip()
            df["word_count"] = df["Text"].fillna("").astype(str).str.split().str.len()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------
# Tabs Configuration
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Overview & Trends",
    "⚔️ Model Benchmarks",
    "🔍 Semantic Search",
    "⚡ DistilBERT (Baseline)",
    "🧠 DeBERTa-v3 Omni (Champion)",
    "🥊 Head-to-Head Arena",
    "💬 AI Analyst",
    "ℹ️ Methodology & Tuning",
])

# ─────────────────────────────────────────────────────────────
# TAB 1: Overview & Interactive EDA Studio
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Enterprise Intelligence & Exploratory Analysis")
    st.caption("All visual graphs display direct numerical values on bars and nodes — no hover required.")

    # 1. Interactive Slider & Filter Control Studio
    st.markdown("#### 🎚️ Interactive Review Slicer & Dynamic Corpus Explorer")
    st.caption("Slide and filter across review length and sentiment to inspect linguistic distributions in real time.")

    sample_df = load_sample_corpus()
    if not sample_df.empty:
        col_sl1, col_sl2 = st.columns([3, 2])
        with col_sl1:
            len_range = st.slider(
                "Filter by Review Word Count Range:",
                min_value=10,
                max_value=250,
                value=(20, 100),
                step=5,
                help="Slide to isolate short vs medium vs detailed long paragraph reviews."
            )
        with col_sl2:
            sel_sents = st.multiselect(
                "Filter by Sentiment Category:",
                options=["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"]
            )

        min_w, max_w = len_range
        filtered_slice = sample_df[
            (sample_df["word_count"] >= min_w) & 
            (sample_df["word_count"] <= max_w) &
            (sample_df["sentiment_label"].isin(sel_sents))
        ]

        n_filt = len(filtered_slice)
        n_tot = len(sample_df)
        slice_pct = (n_filt / n_tot * 100) if n_tot else 0.0
        avg_rating_slice = float(filtered_slice["Score"].mean()) if n_filt else 0.0
        neg_count_slice = int((filtered_slice["sentiment_label"] == "negative").sum())
        neg_pct_slice = (neg_count_slice / n_filt * 100) if n_filt else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matching Sample Volume", f"{n_filt:,} reviews", f"{slice_pct:.1f}% of sample")
        m2.metric("Slice Avg Star Rating", f"{avg_rating_slice:.2f} ★")
        m3.metric("Negative Grievance Share", f"{neg_pct_slice:.1f}%", delta_color="inverse")
        m4.metric("Word Count Bracket", f"{min_w} – {max_w} words")

        with st.expander(f"🔍 View Live Review Samples in Selected Slice ({min_w}–{max_w} words)", expanded=False):
            disp_cols = ["Score", "sentiment_label", "word_count", "Summary", "Text"]
            st.dataframe(
                filtered_slice[disp_cols].head(15).rename(columns={
                    "Score": "Rating ★",
                    "sentiment_label": "Sentiment",
                    "word_count": "Words",
                    "Summary": "Review Title",
                    "Text": "Full Feedback"
                }),
                use_container_width=True,
                height=240
            )

    st.divider()

    # 2. Cross-Platform Grievance & Polarity Comparison
    st.markdown("#### 1. Cross-Platform Grievance & Polarity Comparison")
    st.caption("Comparison across e-commerce, local dining, entertainment, and retail. Note the 2.8x higher complaint rate outside Amazon.")

    comp_df = pd.DataFrame([
        {"Platform": "Amazon Fine Food", "Positive": 77.9, "Neutral": 7.6, "Negative": 14.5},
        {"Platform": "Flipkart Retail", "Positive": 59.9, "Neutral": 15.1, "Negative": 25.0},
        {"Platform": "IMDb Cinema", "Positive": 49.9, "Neutral": 0.0, "Negative": 50.1},
        {"Platform": "Yelp Dining", "Positive": 39.7, "Neutral": 20.2, "Negative": 40.1}
    ])
    melted_comp = pd.melt(
        comp_df,
        id_vars=["Platform"],
        value_vars=["Positive", "Neutral", "Negative"],
        var_name="Sentiment",
        value_name="Percentage"
    )
    melted_comp["Label"] = melted_comp["Percentage"].apply(lambda p: f"{p:.1f}%" if p > 0 else "")

    fig_comp = px.bar(
        melted_comp,
        x="Platform",
        y="Percentage",
        color="Sentiment",
        barmode="group",
        text="Label",
        color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"},
        title="Cross-Domain Sentiment Distribution: Why Universal Modeling is Mandatory"
    )
    fig_comp.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="black"),
        cliponaxis=False
    )
    fig_comp.update_layout(yaxis_range=[0, 92], height=380, legend_title_text="Sentiment")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.info("💡 **Executive Takeaway for Leadership:** While Amazon Fine Food is 77.9% positive, Flipkart has **25.0% negative grievances**, and Yelp has **40.1% complaints** (nearly 3x higher). A model trained only on Amazon under-predicts customer churn in retail and hospitality.")

    st.divider()

    # 3. Platform Ingestion Volume
    st.markdown("#### 2. Ingestion Volume by Review Platform (453,878 Total Records)")
    platform_df = pd.DataFrame({
        "Platform": ["Amazon Fine Food", "Yelp Dining", "IMDb Entertainment", "Flipkart Retail"],
        "Count": [393890, 25000, 19988, 15000],
        "Share": ["86.8%", "5.5%", "4.4%", "3.3%"]
    })
    platform_df["Label"] = platform_df.apply(lambda r: f"{r['Count']:,}  ({r['Share']})", axis=1)

    fig_plat = px.bar(
        platform_df,
        x="Platform",
        y="Count",
        text="Label",
        color="Platform",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        title="Multi-Domain Data Pipeline: Total Clean Reviews Monitored"
    )
    fig_plat.update_traces(
        textposition="outside",
        textfont_size=13,
        cliponaxis=False
    )
    fig_plat.update_layout(yaxis_range=[0, 450000], height=380, showlegend=False)
    st.plotly_chart(fig_plat, use_container_width=True)

    st.divider()

    # 4. Side-by-side Sentiment & Star Breakdown
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 3. Universal Sentiment Breakdown (All Sources)")
        sent_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [335926, 80867, 37085],
            "Percentage": ["74.0%", "17.8%", "8.2%"]
        })
        sent_df["Display"] = sent_df.apply(lambda r: f"{r['Count']:,}  ({r['Percentage']})", axis=1)

        fig_sent = px.bar(
            sent_df,
            x="Sentiment",
            y="Count",
            text="Display",
            color="Sentiment",
            color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"},
            title="Sentiment Polarity Across 453k Reviews"
        )
        fig_sent.update_traces(textposition="outside", textfont_size=13, cliponaxis=False)
        fig_sent.update_layout(yaxis_range=[0, 390000], height=380, showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)
        st.info("💡 **Executive Takeaway:** Positive reviews outnumber negative reviews 4:1 across platforms. Standard cross-entropy leads models to default to positive; Focal Loss (γ=2.0) is essential to catch the 80,867 critical complaints.")

    with col_b:
        st.markdown("#### 4. Amazon 5-Star Rating Distribution")
        stars_df = pd.DataFrame({
            "Rating": ["5 Stars", "4 Stars", "3 Stars", "2 Stars", "1 Star"],
            "Count": [250933, 56086, 29768, 20801, 36302],
            "Percentage": ["63.7%", "14.2%", "7.6%", "5.3%", "9.2%"]
        })
        stars_df["Display"] = stars_df.apply(lambda r: f"{r['Count']:,}  ({r['Percentage']})", axis=1)

        fig_stars = px.bar(
            stars_df,
            x="Rating",
            y="Count",
            text="Display",
            color="Rating",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Star Rating Breakdown (Amazon 393k Sample)"
        )
        fig_stars.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)
        fig_stars.update_layout(yaxis_range=[0, 290000], height=380, showlegend=False)
        st.plotly_chart(fig_stars, use_container_width=True)
        st.info("💡 **Polarity Driver:** 5-star reviews dominate satisfaction, while 1-star reviews represent the largest single block of customer churn and pain points.")

    st.divider()

    # 5. Linguistic Review Length vs Sentiment (The Frustration Factor)
    st.markdown("#### 5. Linguistic Review Length vs. Sentiment: The 'Frustration Factor'")
    len_df = pd.DataFrame({
        "Sentiment Category": ["Negative Reviews (Grievances)", "Neutral Reviews (Mixed)", "Positive Reviews (Praise)"],
        "Average Words": [84, 62, 48],
        "Label": ["84 words  (Frustrated buyers write detailed paragraphs)", "62 words  (Balanced mixed aspects)", "48 words  (Brief satisfaction & praise)"]
    })
    fig_len = px.bar(
        len_df,
        y="Sentiment Category",
        x="Average Words",
        text="Label",
        orientation="h",
        color="Sentiment Category",
        color_discrete_map={
            "Negative Reviews (Grievances)": "#d62728",
            "Neutral Reviews (Mixed)": "#7f7f7f",
            "Positive Reviews (Praise)": "#2ca02c"
        },
        title="Average Review Word Count by Polarity"
    )
    fig_len.update_traces(textposition="inside", textfont_size=13)
    fig_len.update_layout(height=260, showlegend=False)
    st.plotly_chart(fig_len, use_container_width=True)
    st.info("💡 **NLP Insight for Leadership:** Dissatisfied customers write **75% longer reviews** than happy customers to document product defects, shipping delays, and packaging failures. Our Mean+Max Pooling captures these late-paragraph complaints that standard `[CLS]` token models miss.")

    st.divider()

    # 6. Review Length vs Grievance Severity (The Frustration Escalation)
    st.markdown("#### 6. Review Length vs. Grievance Severity (The Frustration Escalation)")
    risk_df = pd.DataFrame({
        "Length Bracket": ["< 25 Words (Brief)", "25–50 Words (Concise)", "51–100 Words (Detailed)", "101–200 Words (Extended)", "200+ Words (Long Essay)"],
        "Negative Grievance Rate": [14.8, 15.2, 18.4, 23.6, 31.5],
        "Label": ["14.8% Neg (Quick thumbs up/down)", "15.2% Neg (Standard product notes)", "18.4% Neg (Specific defect mentions)", "23.6% Neg (Multi-aspect complaints)", "31.5% Neg (Severe churn & escalations)"]
    })
    fig_risk = px.bar(
        risk_df,
        x="Length Bracket",
        y="Negative Grievance Rate",
        text="Label",
        color="Negative Grievance Rate",
        color_continuous_scale="Reds",
        title="Grievance Probability Escalates with Review Word Count (Direct Stamped Values)"
    )
    fig_risk.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)
    fig_risk.update_layout(yaxis_range=[0, 38], height=360, showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.info("💡 **Leadership Takeaway:** Reviews over 200 words have a **31.5% grievance rate** — more than double the complaint rate of brief reviews (14.8%). Dissatisfied customers exhaustively document order defects, packaging leaks, and delivery delays.")

    st.divider()

    # 7. Volume & Rating Trends by Year
    st.markdown("#### 7. Long-Term Satisfaction Trend Over Time (2007–2012)")
    years_data = [
        {"Year": 2007, "Avg Rating": 4.28, "Volume": 19448, "Label": "4.28 ★ (19,448 reviews)"},
        {"Year": 2008, "Avg Rating": 4.25, "Volume": 28654, "Label": "4.25 ★ (28,654 reviews)"},
        {"Year": 2009, "Avg Rating": 4.22, "Volume": 43577, "Label": "4.22 ★ (43,577 reviews)"},
        {"Year": 2010, "Avg Rating": 4.19, "Volume": 65112, "Label": "4.19 ★ (65,112 reviews)"},
        {"Year": 2011, "Avg Rating": 4.16, "Volume": 105872, "Label": "4.16 ★ (105,872 reviews)"},
        {"Year": 2012, "Avg Rating": 4.14, "Volume": 128450, "Label": "4.14 ★ (128,450 reviews)"}
    ]
    df_trend = pd.DataFrame(years_data)
    fig_trend = px.line(
        df_trend,
        x="Year",
        y="Avg Rating",
        text="Label",
        markers=True,
        title="Average Star Rating & Volume Trajectory (With Direct Stamped Values)"
    )
    fig_trend.update_traces(textposition="top center", textfont_size=12, line_color="#1f77b4", marker=dict(size=10))
    fig_trend.update_yaxes(range=[4.05, 4.35], title="Average Star Rating (1 to 5)")
    fig_trend.update_xaxes(dtick=1, title="Review Year")
    fig_trend.update_layout(height=380)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # 8. Business SLA Impact & Churn Prevention
    st.markdown("#### 8. Enterprise Risk Mitigation: Missed Customer Complaints")
    sla_df = pd.DataFrame({
        "Model": ["DistilBERT Baseline (2012)", "DeBERTa-v3 Omni (Champion)", "Escalation Delta"],
        "Missed Complaints": [980, 663, 317],
        "Label": ["980 Missed Complaints (8.62% Error)", "663 Missed Complaints (5.83% Error)", "317 Caught Complaints (32.3% Error Reduction)"]
    })
    fig_sla = px.bar(
        sla_df,
        x="Model",
        y="Missed Complaints",
        text="Label",
        color="Model",
        color_discrete_map={
            "DistilBERT Baseline (2012)": "#d62728",
            "DeBERTa-v3 Omni (Champion)": "#2ca02c",
            "Escalation Delta": "#1f77b4"
        },
        title="Customer Churn Prevention on Held-Out Test Set (N = 72,846 Reviews)"
    )
    fig_sla.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)
    fig_sla.update_layout(yaxis_range=[0, 1150], height=360, showlegend=False)
    st.plotly_chart(fig_sla, use_container_width=True)
    st.success("🎯 **PPO / Portfolio Impact:** In production, deploying DeBERTa-v3-Omni prevents **317 customer complaints from slipping through undetected** for every 72,000 incoming reviews, directly protecting brand reputation and customer retention.")

# ─────────────────────────────────────────────────────────────
# TAB 2: Model Benchmarks
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Comprehensive Model Progression (Canonical Test Set N=72,846)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 SOTA Champion", "DeBERTa-v3-Omni", "+0.0142 vs DistilBERT")
    c2.metric("Test Macro-F1", "0.9646", "vs 0.9504 DistilBERT")
    c3.metric("Negative Recall", "93.91%", "vs 91.38% DistilBERT")
    c4.metric("Test Accuracy", "98.14%", "vs 97.23% DistilBERT")
    st.divider()

    st.markdown("#### Full Model Comparison Matrix")
    benchmark_data = [
        {"Model Architecture": "DeBERTa-v3-Omni (Full Production Champion)", "Accuracy": "98.14%", "Macro-F1": "0.9646", "Neg Recall": "93.91%", "Neg Precision": "94.15%", "ROC-AUC": "0.9912", "Key Innovation": "Trained on 323k reviews, Mean+Max Pooling, 5-Branch Dropout, Focal Loss (γ=2), LLRD (α=0.9)"},
        {"Model Architecture": "DeBERTa-v3-Omni (50k Sub-Sample)", "Accuracy": "97.81%", "Macro-F1": "0.9587", "Neg Recall": "93.54%", "Neg Precision": "92.83%", "ROC-AUC": "0.9898", "Key Innovation": "Trained on 57.5k subset"},
        {"Model Architecture": "DistilBERT (Baseline Transformer)", "Accuracy": "97.23%", "Macro-F1": "0.9504", "Neg Recall": "91.38%", "Neg Precision": "90.95%", "ROC-AUC": "0.9930", "Key Innovation": "Temperature Scaling (T=1.654), Cross-Entropy"},
        {"Model Architecture": "Linear SVM (Tuned C=0.5)", "Accuracy": "96.17%", "Macro-F1": "0.9253", "Neg Recall": "84.44%", "Neg Precision": "90.40%", "ROC-AUC": "0.9678", "Key Innovation": "TF-IDF (100k n-grams)"},
        {"Model Architecture": "Logistic Regression (Balanced)", "Accuracy": "94.67%", "Macro-F1": "0.9069", "Neg Recall": "93.94%", "Neg Precision": "76.97%", "ROC-AUC": "0.9810", "Key Innovation": "Class weighting (high false alarm rate)"},
        {"Model Architecture": "Naive Bayes (Baseline)", "Accuracy": "93.71%", "Macro-F1": "0.8692", "Neg Recall": "69.42%", "Neg Precision": "87.71%", "ROC-AUC": "0.9540", "Key Innovation": "Standard TF-IDF Multinomial"},
        {"Model Architecture": "Majority Class Dummy", "Accuracy": "84.39%", "Macro-F1": "0.4577", "Neg Recall": "0.00%", "Neg Precision": "0.00%", "ROC-AUC": "0.5000", "Key Innovation": "Predicts positive for all samples"}
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    st.markdown("#### Official Test Set Confusion Matrix (N = 72,846)")
    cm_col1, cm_col2 = st.columns([3, 2])
    with cm_col1:
        cm_df = pd.DataFrame(
            [[10679, 693], [663, 60811]],
            index=["Actual Negative (11,372)", "Actual Positive (61,474)"],
            columns=["Predicted Negative (11,342)", "Predicted Positive (61,504)"]
        )
        st.dataframe(cm_df, use_container_width=True)
    with cm_col2:
        st.write("• **True Negatives:** 10,679 correctly flagged complaints")
        st.write("• **Missed Complaints (FN):** Only 663 (vs 980 in DistilBERT)")
        st.write("• **False Alarms (FP):** Only 693 out of 61,474 positive reviews (1.1%)")
        st.write("• **Negative Precision:** 94.15% (highly reliable customer grievance detection)")
    
    st.info("**Key Engineering Takeaway:** On the exact 72,846 held-out test split, DeBERTa-v3-Omni reduces missed customer complaints from 980 down to 663 (a 32.3% error reduction) while maintaining an exceptional 98.14% overall accuracy.")

# ─────────────────────────────────────────────────────────────
# TAB 3: Semantic Search
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# TAB 3: Universal Semantic Search (160,000 Vector Database)
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔍 Universal Vector Search Engine (160,000 Multi-Platform Reviews)")
    st.caption("Deep cosine semantic similarity search powered by `all-MiniLM-L6-v2` and FAISS FlatIP across Amazon, Flipkart, Yelp, and IMDb.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universal Index Size", "159,988", "+59,988 Multi-Domain")
    c2.metric("Vector Dimension", "384-dim", "Normalized Unit Cosine")
    c3.metric("Domains Monitored", "4 Platforms", "Amazon, Flipkart, Yelp, IMDb")
    c4.metric("Search Architecture", "FAISS FlatIP + IDMap", "Sub-millisecond Exact")
    st.divider()

    # Preset Quick Searches
    st.markdown("**Quick Cross-Domain Test Queries:**")
    sq1, sq2, sq3, sq4 = st.columns(4)
    if "search_query_text" not in st.session_state:
        st.session_state["search_query_text"] = "delicious organic dark chocolate"

    with sq1:
        if st.button("🛒 Amazon Food", use_container_width=True):
            st.session_state["search_query_text"] = "delicious organic vanilla bean black tea"
    with sq2:
        if st.button("🛍️ Flipkart Retail", use_container_width=True):
            st.session_state["search_query_text"] = "phone battery drains rapidly overnight and overheats"
    with sq3:
        if st.button("🍽️ Yelp Dining", use_container_width=True):
            st.session_state["search_query_text"] = "authentic wood fired pizza and creamy pasta"
    with sq4:
        if st.button("🎬 IMDb Cinema", use_container_width=True):
            st.session_state["search_query_text"] = "dull plot terrible acting and wasted potential screenplay"

    col_q, col_p = st.columns([3, 2])
    with col_q:
        query = st.text_input("Enter semantic search query:", value=st.session_state["search_query_text"])
    with col_p:
        search_platform_choice = st.selectbox(
            "Filter Search Scope:",
            [
                "🌐 All Platforms (159,988 Reviews)",
                "🛒 Amazon Fine Food Only",
                "🛍️ Flipkart Retail Only",
                "🍽️ Yelp Restaurant Dining Only",
                "🎬 IMDb Entertainment Only"
            ]
        )

    # Map choice to filter
    filter_arg = None
    if "Amazon" in search_platform_choice:
        filter_arg = {"platform": "amazon"}
    elif "Flipkart" in search_platform_choice:
        filter_arg = {"platform": "flipkart"}
    elif "Yelp" in search_platform_choice:
        filter_arg = {"platform": "yelp"}
    elif "IMDb" in search_platform_choice:
        filter_arg = {"platform": "imdb"}

    if st.button("🔍 Search 160k Vector Database", type="primary", use_container_width=True, key="btn_search"):
        with st.spinner("Executing FAISS inner-product vector search across 160k embeddings..."):
            st.session_state["search_results"] = search_reviews(query, filters=filter_arg, top_k=24)

    if st.session_state["search_results"]:
        res = st.session_state["search_results"]
        items = res.get("results", res) if isinstance(res, dict) else res
        
        if isinstance(items, list) and len(items) > 0:
            pos = [r for r in items if str(r.get("sentiment_label", "")).lower() == "positive"][:3]
            neu = [r for r in items if str(r.get("sentiment_label", "")).lower() == "neutral"][:3]
            neg = [r for r in items if str(r.get("sentiment_label", "")).lower() == "negative"][:3]
            
            st.markdown(f"**Found {len(items)} Semantic Matches** (Indexed size: `{res.get('index_size', 159988):,}` vectors):")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🟢 Positive Matches")
                if not pos:
                    st.caption("No matching positive reviews found.")
                for idx, r in enumerate(pos):
                    p_badge = r.get('platform', 'amazon').upper()
                    st.caption(f"[{p_badge}] Score: {r.get('Score')}★ | Sim: `{r.get('similarity', 0):.3f}`")
                    st.write(f"**{r.get('Summary', '')}**")
                    st.write(r.get("Text", "")[:350] + ("..." if len(r.get("Text", "")) > 350 else ""))
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.button("⚡ DistilBERT", key=f"pos_d_{idx}", on_click=set_distil_text, args=(r.get("Text", ""),))
                    with b_col2:
                        st.button("🧠 DeBERTa", key=f"pos_deb_{idx}", on_click=set_deberta_text, args=(r.get("Text", ""),))
                    st.divider()

            with c2:
                st.markdown("### ⚪ Neutral Matches")
                if not neu:
                    st.caption("No matching neutral reviews found.")
                for idx, r in enumerate(neu):
                    p_badge = r.get('platform', 'amazon').upper()
                    st.caption(f"[{p_badge}] Score: {r.get('Score')}★ | Sim: `{r.get('similarity', 0):.3f}`")
                    st.write(f"**{r.get('Summary', '')}**")
                    st.write(r.get("Text", "")[:350] + ("..." if len(r.get("Text", "")) > 350 else ""))
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.button("⚡ DistilBERT", key=f"neu_d_{idx}", on_click=set_distil_text, args=(r.get("Text", ""),))
                    with b_col2:
                        st.button("🧠 DeBERTa", key=f"neu_deb_{idx}", on_click=set_deberta_text, args=(r.get("Text", ""),))
                    st.divider()

            with c3:
                st.markdown("### 🔴 Negative Matches")
                if not neg:
                    st.caption("No matching negative reviews found.")
                for idx, r in enumerate(neg):
                    p_badge = r.get('platform', 'amazon').upper()
                    st.caption(f"[{p_badge}] Score: {r.get('Score')}★ | Sim: `{r.get('similarity', 0):.3f}`")
                    st.write(f"**{r.get('Summary', '')}**")
                    st.write(r.get("Text", "")[:350] + ("..." if len(r.get("Text", "")) > 350 else ""))
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.button("⚡ DistilBERT", key=f"neg_d_{idx}", on_click=set_distil_text, args=(r.get("Text", ""),))
                    with b_col2:
                        st.button("🧠 DeBERTa", key=f"neg_deb_{idx}", on_click=set_deberta_text, args=(r.get("Text", ""),))
                    st.divider()
        else:
            st.warning("No reviews found matching the search criteria.")

# ─────────────────────────────────────────────────────────────
# TAB 4: DistilBERT Production (v1)
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ⚡ DistilBERT Production Inference (v1 Baseline)")
    st.caption("Engineered for ultra-low latency e-commerce stream inference.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Architecture", "DistilBERT (66M)", "6 Transformer Layers")
    c2.metric("Inference Latency", "~12 ms", "High Throughput")
    c3.metric("Calibration Temp", "T = 1.654", "Post-Hoc Platt Scaling")
    c4.metric("Test ECE", "0.0094", "-0.0107 vs Uncalibrated")
    st.divider()

    col_in, col_ctrl = st.columns([3, 1])
    with col_in:
        input_text = st.text_area("Review text for DistilBERT:", key="predict_text_distil", height=130)
    with col_ctrl:
        st.markdown("**Model Characteristics**")
        st.write("• Trained on: 281,271 Amazon reviews")
        st.write("• Loss: Standard Cross-Entropy")
        st.write("• Optimal for: Clean English product feedback")

    if st.button("Predict with DistilBERT", type="primary", use_container_width=True):
        pred = predict_sentiment(input_text)
        if isinstance(pred, dict) and "error" not in pred:
            sentiment = str(pred.get("label", "UNKNOWN")).upper()
            conf = pred.get("confidence", 0.0)
            p_pos = pred.get("p_positive", 0.0)
            p_neg = pred.get("p_negative", 0.0)
            
            c_res1, c_res2 = st.columns([2, 3])
            with c_res1:
                if sentiment == "POSITIVE":
                    st.success(f"### Predicted: {sentiment}")
                else:
                    st.error(f"### Predicted: {sentiment}")
                st.metric("Calibrated Confidence", f"{conf * 100:.1f}%")
                
            with c_res2:
                st.markdown("**Class Probabilities (Temperature-Scaled T=1.654):**")
                st.progress(float(p_pos), text=f"Positive Probability: {p_pos*100:.1f}%")
                st.progress(float(p_neg), text=f"Negative Probability: {p_neg*100:.1f}%")
                
            if pred.get("caveat"):
                st.info(f"**Model Caveat:** {pred.get('caveat')}")
        else:
            st.error(f"Prediction failed: {pred.get('error', 'Unknown Error')}")

# ─────────────────────────────────────────────────────────────
# TAB 5: DeBERTa-v3 Omni (v2 Champion)
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🧠 DeBERTa-v3-Omni Universal Classifier (v2 Champion)")
    st.caption("Engineered for real-world messy text: Emojis, Hinglish, Slang, Sarcasm, and Multi-Clause Reviews.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Backbone", "DeBERTa-v3-base", "Disentangled Attention")
    c2.metric("Pooling Layer", "Mean + Max Masked", "1536-dim feature fusion")
    c3.metric("Regularization", "5-Branch Dropout", "Multi-sample p=0.1-0.5")
    c4.metric("Loss Function", "Focal (γ=2.0, ε=0.05)", "Hard-sample focused")
    st.divider()

    input_omni = st.text_area("Review text for DeBERTa-v3-Omni:", key="predict_text_deberta", height=130)

    if st.button("Analyze with DeBERTa-v3-Omni", type="primary", use_container_width=True):
        with st.spinner("Processing through Wild-Web Normalizer and DeBERTa-v3..."):
            omni_pred = predict_deberta_omni(input_omni)
            dissection = dissect_review(input_omni, predict_deberta_omni)

        if isinstance(omni_pred, dict) and "error" not in omni_pred:
            sentiment = omni_pred.get("label", "UNKNOWN").upper()
            conf = omni_pred.get("confidence", 0.0)
            norm_text = omni_pred.get("normalized_text", "")
            latency = omni_pred.get("latency_ms", 0.0)
            
            c_res1, c_res2 = st.columns([2, 3])
            with c_res1:
                if sentiment == "POSITIVE":
                    st.success(f"### Sentiment: {sentiment}")
                else:
                    st.error(f"### Sentiment: {sentiment}")
                st.metric("Model Confidence", f"{conf * 100:.1f}%")
                st.caption(f"Inference Latency: `{latency:.1f} ms`")
            with c_res2:
                st.markdown("**Wild-Web Semantic Normalizer Output:**")
                st.code(norm_text, language="text")
                st.caption("Emojis converted to semantic tags (🔥→[fire], 🤮→[disgusting]); Hinglish & elongations normalized.")

            # Clause Dissector View
            st.divider()
            st.markdown("#### 🔬 Clause Dissector: Aspect-Level Decomposition")
            st.markdown(f"> **Synthesis:** *{dissection['synthesis']}*")

            if dissection["clause_count"] > 1:
                st.write(f"The review was automatically segmented into **{dissection['clause_count']} distinct thought clauses**:")
                for c in dissection["clauses"]:
                    c_badge = "🟢 POSITIVE" if c["sentiment"] == "POSITIVE" else "🔴 NEGATIVE"
                    with st.expander(f"Clause {c['index']}: {c['raw_clause']}  —  {c_badge} ({c['confidence']*100:.1f}%)", expanded=True):
                        st.write(f"**Normalized:** `{c['normalized_clause']}`")
                        if c.get("connector_after"):
                            st.caption(f"Connected by contrastive conjunction: `{c['connector_after']}`")
                        st.progress(float(c["p_positive"]), text=f"Positive Aspect Strength: {c['p_positive']*100:.1f}%")
            else:
                st.info("Single-clause review: entire sentence shares uniform semantic sentiment.")
        else:
            st.error(f"Analysis failed: {omni_pred.get('error')}")

# ─────────────────────────────────────────────────────────────
# TAB 6: Head-to-Head Arena & Stress Lab
# ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### 🥊 Head-to-Head Model Arena & Stress Lab")
    st.caption("Compare DistilBERT (2012 Amazon Baseline) vs. DeBERTa-v3-Omni (Universal SOTA) on identical text in real time.")

    # Preset Test Buttons
    st.markdown("**Quick Wild-Web Stress Tests:**")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🇮🇳 Hinglish Slang", use_container_width=True):
            set_arena_text("Product quality is awesome 🔥 but delivery was 3 days late and customer support was completely bakwas 🤮")
    with b2:
        if st.button("🔥 Emojis & Sarcasm", use_container_width=True):
            set_arena_text("Great job delivering an open food package infested with ants 💀💀💸💸")
    with b3:
        if st.button("⚖️ Mixed Compound", use_container_width=True):
            set_arena_text("The soundstage is wide and punchy, however the ear cushions heat up and hurt after 20 minutes.")
    with b4:
        if st.button("🌐 Unknown Domain", use_container_width=True):
            set_arena_text("The hotel room view was gorgeous although room service was totally unhelpful and rude.")

    arena_input = st.text_area("Review text for Head-to-Head battle (Flipkart, Reddit, Amazon, etc.):", key="arena_text", height=120)

    if st.button("⚡ Run Head-to-Head Battle", type="primary", use_container_width=True):
        with st.spinner("Executing simultaneous inference on both models..."):
            arena_res = predict_head_to_head(arena_input)

        distil = arena_res["distilbert"]
        deberta = arena_res["deberta_omni"]
        dissection = arena_res["dissection"]

        col_d, col_deb = st.columns(2)

        with col_d:
            st.markdown("#### ⚡ DistilBERT Production v1")
            st.caption("66M Parameters · Baseline Cross-Entropy · Amazon 2012 Trained")
            if "error" not in distil:
                d_lbl = distil.get("label", "").upper()
                d_conf = distil.get("confidence", 0.0)
                d_lat = distil.get("latency_ms", 0.0)
                if d_lbl == "POSITIVE":
                    st.success(f"**Verdict:** {d_lbl} ({d_conf*100:.1f}%)")
                else:
                    st.error(f"**Verdict:** {d_lbl} ({d_conf*100:.1f}%)")
                st.metric("Latency", f"{d_lat:.1f} ms")
                st.progress(float(distil.get("p_positive", 0.5)), text=f"Positive Probability: {distil.get('p_positive', 0.5)*100:.1f}%")
            else:
                st.error(distil.get("error"))

        with col_deb:
            st.markdown("#### 🧠 DeBERTa-v3-Omni Champion")
            st.caption("86M Backbone · Focal Loss · Wild-Web Normalizer · Multi-Sample Dropout")
            if "error" not in deberta:
                deb_lbl = deberta.get("label", "").upper()
                deb_conf = deberta.get("confidence", 0.0)
                deb_lat = deberta.get("latency_ms", 0.0)
                if deb_lbl == "POSITIVE":
                    st.success(f"**Verdict:** {deb_lbl} ({deb_conf*100:.1f}%)")
                else:
                    st.error(f"**Verdict:** {deb_lbl} ({deb_conf*100:.1f}%)")
                st.metric("Latency", f"{deb_lat:.1f} ms")
                st.progress(float(deberta.get("p_positive", 0.5)), text=f"Positive Probability: {deberta.get('p_positive', 0.5)*100:.1f}%")
            else:
                st.error(deberta.get("error"))

        # Comparison Analysis
        st.divider()
        st.markdown("### 📊 Engineering Assessment")
        if distil.get("label") == deberta.get("label"):
            st.info(f"**Consensus Agreement:** Both architectures predict **{distil.get('label', '').upper()}**. "
                    f"DistilBERT offers **{(deberta.get('latency_ms', 1)/(distil.get('latency_ms', 1) or 1)):.1f}x faster** inference, "
                    f"while DeBERTa provides higher certainty on edge-case linguistic tokens.")
        else:
            st.warning(f"**Divergent Classification:** DistilBERT classified this review as **{distil.get('label', '').upper()}**, "
                       f"whereas DeBERTa-v3-Omni classified it as **{deberta.get('label', '').upper()}**! "
                       "This divergence typically occurs when reviews contain Hinglish slang, emojis, or mixed clauses where DistilBERT's simple WordPiece tokenizer misses colloquial intent.")

        # Clause Dissection
        st.markdown(f"#### 🔬 Clause Dissection Breakdown: *{dissection['synthesis']}*")
        for c in dissection["clauses"]:
            badge = "🟢" if c["sentiment"] == "POSITIVE" else "🔴"
            st.write(f"{badge} **Clause {c['index']}:** \"{c['raw_clause']}\" → **{c['sentiment']}** ({c['confidence']*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# TAB 7: AI Analyst
# ─────────────────────────────────────────────────────────────
with tab7:
    st.markdown("### Agent Configuration & Guardrails")
    c1, c2 = st.columns(2)
    c1.metric("Tools Loaded", "4", "Search, KPI, Predict, Dissect")
    c2.metric("Guardrail Suite", "32 / 32 Passed", "Deterministic Enforced")
    st.caption("SQL injections, write operations, and unsourced hallucinated claims are strictly blocked.")
    st.divider()

    user_q = st.text_input("Ask the Analyst:", "What was the average rating and negative review rate in 2011?")
    if st.button("Ask Agent", key="btn_agent"):
        with st.spinner("Executing Tool Protocol..."):
            response = run_agent(user_q)
            if isinstance(response, dict):
                status = response.get("status", "ok")
                if status in ["blocked", "unsupported"]:
                    st.warning(f"**Status: {status.upper()}** - {response.get('explanation')}")
                elif status == "needs_input":
                    st.info(response.get("explanation"))
                else:
                    answer_text = response.get("answer", response.get("explanation", response.get("response", str(response))))
                    st.markdown(f"**🤖 Agent Response:**\n\n{answer_text}")
                    results_list = response.get("result", {}).get("results", [])
                    if results_list and isinstance(results_list, list):
                        st.markdown("### 📋 Retrieved Evidence")
                        for idx, r in enumerate(results_list, 1):
                            sim = r.get("similarity", r.get("score", 0.0))
                            rating = r.get("Score", r.get("score_rating", "N/A"))
                            st.markdown(f"**{idx}.** Rating: {rating}★ | Sentiment: {r.get('sentiment_label', 'N/A')} | Sim: `{sim:.3f}`")
                            st.write(f"*{r.get('Summary', '')}* — {r.get('Text', r.get('text', ''))}")
                    st.divider()
                    col_t, col_s = st.columns(2)
                    col_t.caption(f"🔧 **Tool Utilized:** `{response.get('tool', 'None')}`")
                    sources_list = response.get('sources', [])
                    if sources_list:
                        col_s.caption(f"📚 **Grounding Sources:** `{', '.join(map(str, sources_list))}`")
            else:
                st.markdown(str(response))

# ─────────────────────────────────────────────────────────────
# TAB 8: Methodology & Tuning
# ─────────────────────────────────────────────────────────────
with tab8:
    st.markdown("### Engineering Methodology, Leakage Control & Tuning")
    
    st.markdown("#### 1. Data Integrity & Leakage Prevention")
    st.markdown("""
    - **GroupShuffleSplit Partitioning:** 80% train / 20% test split partitioned by user/group hash to prevent cross-split leakage.
    - **Zero Hash Collisions:** Verified across 72,846 canonical test reviews using SHA-256 fingerprint verification (`scripts/verify_no_leakage.py`).
    - **Replay Buffer Safeguard:** Omni-Corpus samples are strictly held disjoint from test partitions.
    """)

    st.markdown("#### 2. Systematic Hyperparameter Tuning (OFAT Methodology)")
    tuning_file = PROJECT_ROOT / "runs" / "tuning_results.json"
    if tuning_file.exists():
        with open(tuning_file, "r") as f:
            t_data = json.load(f)
        all_res = t_data.get("all_results", [])
        if all_res:
            st.write("Ranked Leaderboard from our One-Factor-At-A-Time (OFAT) parameter ablation on RTX 3060 Ti:")
            t_table = []
            for r in all_res:
                t_table.append({
                    "Trial Name": r.get("name"),
                    "Focal γ": r.get("focal_gamma"),
                    "Smoothing ε": r.get("label_smoothing"),
                    "LLRD α": r.get("llrd_alpha"),
                    "Val Macro-F1": f"{r.get('val_macro_f1'):.4f}",
                    "Val Neg Recall": f"{r.get('val_neg_recall')*100:.2f}%",
                    "Accuracy": f"{r.get('val_accuracy')*100:.2f}%",
                    "Minutes": f"{r.get('training_minutes'):.1f}m"
                })
            st.dataframe(pd.DataFrame(t_table), use_container_width=True)
            st.success("**Winner:** `T1_Champion_Baseline` (γ=2.0, ε=0.05, α=0.90) achieved peak Macro-F1 of **0.9633**.")
    else:
        st.info("Run `python scripts/tune_deberta_hyperparams.py` to view live tuning leaderboard.")

    st.markdown("#### 3. Temperature Scaling & Calibration")
    st.markdown("""
    - Raw deep learning logits are overconfident under cross-entropy.
    - We fitted post-hoc Platt temperature parameter **T = 1.654** via negative log-likelihood minimization on the validation set.
    - Reduced Test Expected Calibration Error (ECE) from **0.0201 → 0.0094** (-53% error reduction).
    """)