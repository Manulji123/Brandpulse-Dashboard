import streamlit as st
import pandas as pd
from pymongo import MongoClient
import certifi
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page Setup 
st.set_page_config(page_title="BrandPulse Dashboard", layout="wide")
st.title("📊 BrandPulse: KFC AI-Powered Brand Analysis")

# VADER Sentiment Helper Function 
def get_vader_sentiment(text, analyzer):
    #"""Calculates VADER sentiment for a given text."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 'neutral'
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Data Loading Function
@st.cache_data(ttl=600)
def load_data():
    print("Connecting to MongoDB Warehouse...")
    try:
        uri = st.secrets["MONGO_URI"]
    except FileNotFoundError:
        st.error("Secrets not found! Make sure you added MONGO_URI to Streamlit Cloud secrets.")
        st.stop()

    mongo_client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
    db = mongo_client["brandpulse"]
    
    # 1. Load Main Records
    print("Loading text records...")
    projection = {"platform": 1, "text": 1, "transcript_text": 1, "brand": 1}
    df_records = pd.DataFrame(list(db["records"].find({}, projection)))
    
    if not df_records.empty:
        analyzer = SentimentIntensityAnalyzer()
        df_records['text'] = df_records['text'].fillna('')
        df_records['transcript_text'] = df_records['transcript_text'].fillna('')
        df_records['analysis_text'] = df_records['text'] + ' ' + df_records['transcript_text']
        df_records['sentiment_vader'] = df_records['analysis_text'].apply(get_vader_sentiment, args=(analyzer,))
    else:
        df_records['sentiment_vader'] = None 

    # 2. Load Visual Analysis
    print("Loading visual analysis...")
    df_visual = pd.DataFrame(list(db["visual_analysis"].find({})))

    # 3. Load AI Text Aspect Analysis 
    print("Loading AI text aspect analysis...")
    df_aspects = pd.DataFrame(list(db["text_aspects"].find({})))

    # 4. Load AI Suggestion Summary
    print("Loading AI text suggestion summary...")
    summary_doc = db["text_suggestions"].find_one()
    ai_suggestions_summary = summary_doc['summary'] if summary_doc else "No AI summary found."
    
    print("Data loading complete.")
    return df_records, df_visual, df_aspects, ai_suggestions_summary

# Main App Body 
if st.button("🔄 Refresh Data from Database"):
    st.cache_data.clear()

try:
    df_records, df_visual, df_aspects, ai_suggestions_summary = load_data()
    
    # 1. Top-Level Text Analysis
    st.header(f"📈 Overall Text Analysis (over {round(len(df_records),-3)} Posts)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mentions by Platform & Sentiment")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.countplot( data=df_records, x='platform', hue='sentiment_vader', palette={'positive': 'g', 'negative': 'r', 'neutral': 'b'}, ax=ax1 )
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Overall Brand Sentiment")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        
        if 'sentiment_vader' in df_records and not df_records.empty:
            sentiment_counts = df_records['sentiment_vader'].value_counts()
            labels = sentiment_counts.index
            colors = ['g' if l == 'positive' else 'r' if l == 'negative' else 'b' for l in labels]
            ax2.pie( sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140 )
            ax2.axis('equal')
            st.pyplot(fig2)
        else:
            st.warning("No sentiment data to display.")

    # 2. AI-Powered Text Insights
    st.header("🧠 AI-Powered Text Insights")
    st.subheader("AI-Generated Strategic Recommendations")
    st.markdown(ai_suggestions_summary)
    
    st.subheader("Sentiment by Key Aspect")
    if not df_aspects.empty:
        unique_aspects = sorted(df_aspects['aspect'].unique())
        num_aspects = len(unique_aspects)
        num_cols = 4 
        num_rows = (num_aspects + num_cols - 1) // num_cols 
        color_map = {'positive': 'g', 'negative': 'r', 'neutral': 'b'}
    
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_aspects:
                    aspect = unique_aspects[idx]
                    with cols[j]:
                        aspect_data = df_aspects[df_aspects['aspect'] == aspect]
                        sentiment_counts = aspect_data['sentiment'].value_counts()
                        labels = sentiment_counts.index
                        colors = [color_map.get(str(label).lower(), 'gray') for label in labels]
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie(sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f"{aspect}")
                        ax.axis('equal')
                        st.pyplot(fig)
    else:
        st.error("No AI aspect data found.")

    # 3. Visual Analysis Summary 
    st.header(f"👁️ Overall Visual Analysis ({round(len(df_visual),-2)} Images)")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.subheader("Most Common Emotions")
        if not df_visual.empty:
            emotions_list = df_visual['emotions'].explode()
            emotions_df = pd.json_normalize(emotions_list)
            emotions_df = emotions_df[emotions_df['dominant_emotion'].notna()]
            st.dataframe(emotions_df['dominant_emotion'].value_counts())
        else:
            st.error("No visual data.")
    
    with col6:
        st.subheader("Most Common Objects")
        if not df_visual.empty:
            all_objects = df_visual['objects'].explode()
            st.dataframe(all_objects[all_objects != "No objects found."].value_counts().head(20))
        else:
            st.error("No visual data.")
    
    with col7:
        st.subheader("Most Common OCR Text")
        if not df_visual.empty:
            all_ocr = df_visual['ocr_text'].explode()
            st.dataframe(all_ocr[all_ocr != "No text found."].value_counts().head(20))
        else:
            st.error("No visual data.")

except Exception as e:
    st.error(f"Dashboard Error: {e}")