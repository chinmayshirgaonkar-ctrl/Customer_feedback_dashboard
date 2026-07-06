import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Configure page layout
st.set_page_config(page_title="Customer Feedback Dashboard", layout="wide")

st.title("📊 Customer Feedback Sentiment Dashboard")
st.markdown("Analyze customer feedback using TextBlob, Streamlit, and Plotly.")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('sample_feedback.csv')

df = load_data()

# Sentiment Analysis Logic
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Process the data
df['Sentiment'] = df['Feedback'].apply(get_sentiment)
df['Polarity Score'] = df['Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Dashboard Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig_bar = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count', 
        color='Sentiment',
        color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Sentiment Proportions")
    fig_pie = px.pie(
        sentiment_counts, 
        names='Sentiment', 
        values='Count',
        color='Sentiment',
        color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Detailed Feedback Data")
st.dataframe(df[['Date', 'Customer', 'Feedback', 'Sentiment', 'Polarity Score']], use_container_width=True)
