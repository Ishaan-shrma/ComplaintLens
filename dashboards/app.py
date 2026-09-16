import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Amazon Fine Food Analytics", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    for path in [
        Path("data/processed/reviews_cleaned.csv"),
        Path("../data/processed/reviews_cleaned.csv"),
        Path("A:/DS-Review-AI/data/processed/reviews_cleaned.csv"),
    ]:
        if path.exists():
            return pd.read_csv(path)
    return None

df = load_data()
if df is None:
    st.error("Could not find reviews_cleaned.csv")
    st.stop()

st.title("Amazon Fine Food Reviews — Dashboard")

# ---------- KPI cards ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Average Rating", f"{df['Score'].mean():.2f} ⭐")
col3.metric("Positive %", f"{(df['sentiment_label']=='positive').mean()*100:.1f}%")
col4.metric("Negative %", f"{(df['sentiment_label']=='negative').mean()*100:.1f}%")

st.markdown("---")

# ---------- Rating trend ----------
st.subheader("Average Rating by Year")
rating_by_year = df.groupby('review_year')['Score'].mean().reset_index()
fig1 = px.line(rating_by_year, x='review_year', y='Score', markers=True)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Sentiment mix ----------
st.subheader("Sentiment Mix")
sentiment_counts = df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig2 = px.pie(sentiment_counts, names='Sentiment', values='Count', hole=0.4)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Review examples ----------
st.subheader("Sample Reviews")
sentiment_filter = st.selectbox("Show reviews:", ['positive', 'negative', 'neutral'])
sample = df[df['sentiment_label'] == sentiment_filter][['ProductId', 'Score', 'Summary', 'Text']].sample(5)
st.dataframe(sample, use_container_width=True, hide_index=True)