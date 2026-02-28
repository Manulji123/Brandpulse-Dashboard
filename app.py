import streamlit as st
import pandas as pd
from pymongo import MongoClient
import certifi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrandPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#3498db"
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def sentiment_pie(ax, counts, title=""):
    labels = counts.index.tolist()
    colors = [SENTIMENT_COLORS.get(str(l).lower(), "#95a5a6") for l in labels]
    ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data():
    uri = st.secrets["MONGO_URI"]
    client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
    db = client["brandpulse"]

    # 1. Raw records (for VADER counts)
    recs = list(db["records"].find(
        {},
        {"platform": 1, "record_type": 1, "brand": 1,
         "text": 1, "transcript_text": 1, "captions_text": 1,
         "source_title": 1, "url": 1, "created_at": 1,
         "ai_overall_sentiment": 1}
    ))
    df_records = pd.DataFrame(recs)
    if not df_records.empty:
        df_records["_id"] = df_records["_id"].astype(str)
        for c in ["text", "transcript_text", "captions_text",
                  "source_title", "platform", "record_type", "brand",
                  "url", "created_at", "ai_overall_sentiment"]:
            if c not in df_records.columns:
                df_records[c] = ""
            df_records[c] = df_records[c].fillna("")

    # 2. AI aspects  (collection written by save-to-mongo cell)
    aspects = list(db["analysis_aspects"].find(
        {},
        {"brand": 1, "platform": 1, "aspect": 1,
         "sentiment": 1, "reason": 1, "record_id": 1,
         "overall_sentiment": 1}
    ))
    df_aspects = pd.DataFrame(aspects)
    if not df_aspects.empty:
        df_aspects["_id"] = df_aspects["_id"].astype(str)
        for c in ["brand", "platform", "aspect", "sentiment",
                  "reason", "record_id", "overall_sentiment"]:
            if c not in df_aspects.columns:
                df_aspects[c] = ""
            df_aspects[c] = df_aspects[c].fillna("")

    # 3. AI suggestions (one-doc summary + individual rows)
    sugg_rows = list(db["analysis_suggestions"].find(
        {}, {"brand": 1, "platform": 1, "suggestion": 1}
    ))
    df_suggestions = pd.DataFrame(sugg_rows)
    if not df_suggestions.empty:
        df_suggestions["_id"] = df_suggestions["_id"].astype(str)

    # 4. Strategic summary text (stored by save cell as a single doc)
    summary_doc = db["analysis_summary"].find_one({})
    ai_summary = summary_doc.get("summary", "") if summary_doc else ""

    client.close()
    return df_records, df_aspects, df_suggestions, ai_summary


# ── App ───────────────────────────────────────────────────────────────────────
st.title("📊 BrandPulse: AI-Powered Brand Analysis Dashboard")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

try:
    df_records, df_aspects, df_suggestions, ai_summary = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Make sure MONGO_URI is set in Streamlit Cloud → App Settings → Secrets")
    st.stop()

brand = df_records["brand"].iloc[0] if not df_records.empty and "brand" in df_records.columns else "Brand"

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")

    all_brands = ["All"] + sorted(df_records["brand"].unique().tolist()) if not df_records.empty else ["All"]
    selected_brand = st.selectbox("Brand", all_brands)

    all_platforms = ["All"] + sorted(df_records["platform"].unique().tolist()) if not df_records.empty else ["All"]
    selected_platform = st.selectbox("Platform", all_platforms)

    st.markdown("---")
    st.caption(f"Total records: {len(df_records):,}")
    st.caption(f"Aspects extracted: {len(df_aspects):,}")
    st.caption(f"Suggestions: {len(df_suggestions):,}")

# Apply filters
df_r = df_records.copy()
df_a = df_aspects.copy()
df_s = df_suggestions.copy()

if selected_brand != "All":
    df_r = df_r[df_r["brand"] == selected_brand]
    df_a = df_a[df_a["brand"] == selected_brand]
    df_s = df_s[df_s["brand"] == selected_brand]

if selected_platform != "All":
    df_r = df_r[df_r["platform"] == selected_platform]
    df_a = df_a[df_a["platform"] == selected_platform]
    df_s = df_s[df_s["platform"] == selected_platform]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Top-level metrics
# ─────────────────────────────────────────────────────────────────────────────
st.header("📈 Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Records",    f"{len(df_r):,}")
m2.metric("Platforms",        df_r["platform"].nunique() if not df_r.empty else 0)
m3.metric("Aspects Found",    f"{len(df_a):,}")
m4.metric("Suggestions",      f"{len(df_s):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Sentiment overview
# ─────────────────────────────────────────────────────────────────────────────
st.header("🎭 Sentiment Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("AI Sentiment — All Records")
    if not df_r.empty and "ai_overall_sentiment" in df_r.columns:
        ai_sent = df_r[df_r["ai_overall_sentiment"] != ""]["ai_overall_sentiment"].value_counts()
        if not ai_sent.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            sentiment_pie(ax, ai_sent)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No AI sentiment data yet. Run analysis notebook.")
    else:
        st.info("No sentiment data.")

with col2:
    st.subheader("Sentiment by Platform")
    if not df_r.empty and "ai_overall_sentiment" in df_r.columns:
        plat_sent = df_r[df_r["ai_overall_sentiment"] != ""].groupby(
            ["platform", "ai_overall_sentiment"]
        ).size().reset_index(name="count")
        if not plat_sent.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            pivot = plat_sent.pivot(index="platform", columns="ai_overall_sentiment", values="count").fillna(0)
            pivot.plot(kind="bar", ax=ax,
                       color=[SENTIMENT_COLORS.get(c, "#95a5a6") for c in pivot.columns],
                       edgecolor="white")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.legend(title="Sentiment", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No platform sentiment data.")
    else:
        st.info("No data.")

with col3:
    st.subheader("Records by Platform")
    if not df_r.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        plat_counts = df_r["platform"].value_counts()
        ax.barh(plat_counts.index, plat_counts.values,
                color=["#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"])
        ax.set_xlabel("Count")
        for i, v in enumerate(plat_counts.values):
            ax.text(v + 0.5, i, str(v), va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — AI Aspect Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.header("🧠 AI-Powered Aspect Analysis")

if df_a.empty:
    st.warning("No aspect data found. Run the Analysis + Save-to-MongoDB notebooks first.")
else:
    # Top aspects bar chart
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("Most Mentioned Aspects")
        top_aspects = df_a["aspect"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(top_aspects.index[::-1], top_aspects.values[::-1], color="#3498db")
        ax.set_xlabel("Mentions")
        for bar, val in zip(bars, top_aspects.values[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Overall Aspect Sentiment")
        overall_sent = df_a["sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        sentiment_pie(ax, overall_sent)
        st.pyplot(fig)
        plt.close()

    # Sentiment per aspect stacked bar
    st.subheader("Sentiment Breakdown per Aspect")
    pivot = df_a.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    # Ensure all sentiment columns present
    for s in ["positive", "negative", "neutral"]:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[["positive", "negative", "neutral"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total").head(15)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[SENTIMENT_COLORS["positive"],
                      SENTIMENT_COLORS["negative"],
                      SENTIMENT_COLORS["neutral"]],
               edgecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-aspect pie charts grid
    st.subheader("Sentiment per Aspect (Detailed)")
    unique_aspects = sorted(df_a["aspect"].unique())
    num_cols = 4
    rows = [unique_aspects[i:i+num_cols] for i in range(0, len(unique_aspects), num_cols)]
    for row_aspects in rows:
        cols = st.columns(num_cols)
        for col, aspect in zip(cols, row_aspects):
            with col:
                aspect_data = df_a[df_a["aspect"] == aspect]["sentiment"].value_counts()
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                sentiment_pie(ax, aspect_data, title=aspect)
                st.pyplot(fig)
                plt.close()

    # Top negative reasons
    st.subheader("🔴 Top Negative Reasons by Aspect")
    neg = df_a[df_a["sentiment"] == "negative"][["aspect", "reason"]]
    if not neg.empty:
        aspect_filter = st.selectbox(
            "Filter by aspect",
            ["All"] + sorted(neg["aspect"].unique().tolist())
        )
        if aspect_filter != "All":
            neg = neg[neg["aspect"] == aspect_filter]
        st.dataframe(
            neg[["aspect", "reason"]].head(20).reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("No negative aspects found.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Strategic Recommendations
# ─────────────────────────────────────────────────────────────────────────────
st.header("💡 AI-Generated Strategic Recommendations")

if ai_summary:
    st.markdown(ai_summary)
elif not df_s.empty:
    st.info("No summarized recommendation found. Showing raw suggestions:")
    st.dataframe(
        df_s[["platform", "suggestion"]].head(20).reset_index(drop=True),
        use_container_width=True
    )
else:
    st.warning(
        "No recommendations found. Run the Analysis notebook then the "
        "Save-to-MongoDB notebook to populate this section."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Platform deep-dive
# ─────────────────────────────────────────────────────────────────────────────
st.header("🔎 Platform Deep-Dive")

tabs = st.tabs(["Twitter / X", "Reddit", "YouTube", "Google Maps"])
platform_map = {
    "Twitter / X":  "twitter",
    "Reddit":       "reddit",
    "YouTube":      "youtube",
    "Google Maps":  "google_maps",
}

for tab, (tab_label, plat_key) in zip(tabs, platform_map.items()):
    with tab:
        df_p = df_r[df_r["platform"] == plat_key]
        df_pa = df_a[df_a["platform"] == plat_key]

        if df_p.empty:
            st.info(f"No {tab_label} data collected yet.")
            continue

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Records", len(df_p))
            rt_counts = df_p["record_type"].value_counts()
            st.dataframe(rt_counts.reset_index().rename(
                columns={"index": "type", "record_type": "count"}
            ), use_container_width=True)

        with c2:
            if not df_pa.empty:
                st.subheader("Top Aspects")
                top_p = df_pa["aspect"].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(top_p.index[::-1], top_p.values[::-1], color="#9b59b6")
                ax.set_xlabel("Mentions")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No aspect data for this platform.")

        # Sample records table
        st.subheader(f"Sample {tab_label} Records")
        show_cols = [c for c in ["record_type", "source_title", "text", "url", "created_at"]
                     if c in df_p.columns]
        st.dataframe(
            df_p[show_cols].head(10).reset_index(drop=True),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Raw data explorer
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗃️ Raw Data Explorer"):
    data_choice = st.radio(
        "View dataset:",
        ["Records", "Aspects", "Suggestions"],
        horizontal=True
    )
    if data_choice == "Records":
        st.dataframe(df_r.head(100), use_container_width=True)
    elif data_choice == "Aspects":
        st.dataframe(df_a.head(100), use_container_width=True)
    else:
        st.dataframe(df_s.head(100), use_container_width=True)

st.caption("BrandPulse — Powered by DeepSeek AI + MongoDB Atlas")
