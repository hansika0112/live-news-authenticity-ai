# 🧠 Live News Authenticity Detector - Streamlit App
import streamlit as st
import requests
import joblib
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_fake_real(news_text):
    cleaned = clean_text(news_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    confidence = max(prob) * 100
    label = "🛑 Fake News" if prediction == 1 else "✅ Real News"
    return f"{label} ({confidence:.2f}%)"

# Streamlit UI
st.set_page_config(page_title="Live News Authenticity Detector", page_icon="🧠")
st.title("🧠 Live News Authenticity Detector")
st.markdown("Check if a live news headline is *Real* or *Fake* using AI.")

# Option to fetch live headlines
st.subheader("🔍 Fetch and Check Live News")
query = st.text_input("Enter keyword (eg: India, election, sports)", "India")

if st.button("Get and Detect"):
    from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=10&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "articles" in data and len(data["articles"]) > 0:
        for article in data["articles"]:
            title = article.get("title", "")
            if title:
                result = predict_fake_real(title)
                st.markdown(f"**{title}** → {result}")
    else:
        st.error("No articles found. Try a different keyword.")

# Option to enter your own headline
st.subheader("📝 Try Your Own Headline")
user_input = st.text_input("Enter news headline")
if st.button("Check Headline"):
    if user_input.strip() == "":
        st.warning("Please enter a headline.")
    else:
        result = predict_fake_real(user_input)
        st.success(f"Prediction: {result}")
