import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from transformers import pipeline

# Unduh stopwords jika belum
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# === Load model dan vectorizer yang sudah dilatih ===
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "naive_bayes_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# === Load summarizer ===
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

# === Preprocessing sama seperti di file utama ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# === Summarizer dinamis ===
def summarize_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    if len(text.split()) <= 10:
        return text
    max_len = min(75, int(len(text.split()) * 2))
    min_len = max(15, int(len(text.split()) * 0.6))
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

# === Prediksi dan ringkasan ===
def predict_sentiment_and_summary(text):
    cleaned = clean_text(text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    summary = summarize_text(text)
    return summary, prediction

# === UI Streamlit ===
st.set_page_config(page_title="Analisis Sentimen Judi Online", layout="centered")

st.title("🎲 Analisis Sentimen Judol + Ringkasan Otomatis")

input_text = st.text_area("Masukkan teks sentimen judol di sini 👇", height=200)

if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        summary, prediction = predict_sentiment_and_summary(input_text)

        # Warna dinamis sesuai sentimen
        if prediction == "Positif":
            color = "#FF4B4B"  # merah
        else:
            color = "#2EC4B6"  # biru kehijauan

        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 20px; border-radius: 10px; color: white;">
                <h4>Ringkasan Teks:</h4>
                <p>{summary}</p>
                <h4>Hasil Prediksi Sentimen:</h4>
                <p style="font-size:24px;"><strong>{prediction}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
