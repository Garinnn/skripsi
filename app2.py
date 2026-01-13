import streamlit as st
import joblib
import re
import string
import time
import numpy as np

# ======= Fungsi Pembersihan Teks =======
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ======= Load Model & Vectorizer =======
@st.cache_resource
def load_models():
    model_berita = joblib.load("model_hoax_berita.pkl")
    vectorizer_berita = joblib.load("tfidf_vectorizer_berita.pkl")
    model_sosmed = joblib.load("model_hoax_sosmed.pkl")
    vectorizer_sosmed = joblib.load("tfidf_vectorizer_sosmed.pkl")
    return model_berita, vectorizer_berita, model_sosmed, vectorizer_sosmed


mnb_berita, vec_berita, mnb_sosmed, vec_sosmed = load_models()


# ======= Setup UI (Dark Mode Style) =======
st.set_page_config(page_title="Deteksi Hoax Indonesia", layout="wide")

# ======= CSS DARK MODE PUTIH =======
st.markdown("""
<style>
/* ===== Background & Font Global ===== */
body {
    background-color: #0f1116;
    color: #ffffff;
    font-family: "Poppins", sans-serif;
}
div[data-testid="stMarkdownContainer"] p, span, label {
    color: #ffffff !important;
}
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* ===== Tabs Styling ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1e1f25;
    color: #ffffff;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}
.stTabs [aria-selected="true"] {
    background-color: #007bff !important;
    color: #ffffff !important;
    font-weight: 600;
}

/* ===== Text Area ===== */
textarea {
    background-color: #1a1c22 !important;
    color: white !important;
    border: 1px solid #007bff !important;
    border-radius: 8px !important;
}

/* ===== Button ===== */
div.stButton > button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0056b3;
}

/* ===== Metric & Progress ===== */
div[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: bold;
}
.stProgress > div > div > div {
    background-color: #007bff !important;
}

/* ===== Highlight Kata ===== */
span.hoax {
    background-color: #ff4d4d;
    color: white;
    padding: 2px 5px;
    border-radius: 3px;
}
span.real {
    background-color: #28a745;
    color: white;
    padding: 2px 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)


# ======= Header =======
st.title("🧠 Identifikasi Berita Hoax")
st.markdown("""
Aplikasi ini mendeteksi apakah sebuah **berita** atau **postingan media sosial**
termasuk *HOAX* atau *REAL* menggunakan **Multinomial Naive Bayes (MNB)**.
""")


# ======= Fungsi Highlight Kata =======
def highlight_text(input_text, vectorizer, model, top_n=10, aktifkan_highlight=False):
    cleaned = clean_text(input_text)
    prob = model.predict_proba(vectorizer.transform([cleaned]))[0]
    pred = model.predict(vectorizer.transform([cleaned]))[0]

    if not aktifkan_highlight:
        return cleaned, prob, pred

    feature_names = vectorizer.get_feature_names_out()
    log_probs = model.feature_log_prob_
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    weights = log_probs[pred]

    scores = {w: weights[feature_names.tolist().index(w)]
              for w in words if w in feature_names}

    top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    highlighted = []
    for w in words:
        if w in dict(top_words):
            color_class = "hoax" if pred == 1 else "real"
            highlighted.append(f"<span class='{color_class}'>{w}</span>")
        else:
            highlighted.append(w)

    return " ".join(highlighted), prob, pred


# ======= Fungsi Menampilkan Hasil =======
def tampilkan_hasil(prob, pred, teks_asli, teks_highlight):
    label = "REAL ✅" if pred == 0 else "HOAX ❌"
    st.markdown("---")
    st.subheader("Hasil Deteksi")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Prediksi", label)
    with col2:
        st.progress(prob[1] if pred == 1 else prob[0])

    st.markdown("### 🔍 Kata yang paling berpengaruh:")
    st.markdown(f"<p style='font-size:18px'>{teks_highlight}</p>", unsafe_allow_html=True)
    st.caption("🔴 Merah: kata yang meningkatkan kemungkinan HOAX, 🟢 Hijau: mendukung REAL.")


# ======= Tabs =======
tab1, tab2 = st.tabs(["📰 Analisis Berita", "💬 Analisis Sosial Media"])

# ======= TAB 1 =======
with tab1:
    st.subheader("Masukkan teks berita:")
    teks_berita = st.text_area("", height=180, placeholder="Tempel teks berita di sini...")

    if st.button("Deteksi Berita", use_container_width=True):
        if teks_berita.strip():
            with st.spinner("Menganalisis teks..."):
                time.sleep(1)
                teks_highlight, prob, pred = highlight_text(teks_berita, vec_berita, mnb_berita)
                tampilkan_hasil(prob, pred, teks_berita, teks_highlight)
        else:
            st.error("⚠️ Masukkan teks berita terlebih dahulu.")


# ======= TAB 2 =======
with tab2:
    st.subheader("Masukkan teks postingan media sosial:")
    teks_sosmed = st.text_area("", height=180, key="sosmed_input", placeholder="Tulis postingan sosmed di sini...")

    if st.button("Deteksi Sosial Media", use_container_width=True):
        if teks_sosmed.strip():
            with st.spinner("Menganalisis teks..."):
                time.sleep(1)
                teks_highlight, prob, pred = highlight_text(teks_sosmed, vec_sosmed, mnb_sosmed)
                tampilkan_hasil(prob, pred, teks_sosmed, teks_highlight)
        else:
            st.error("⚠️ Masukkan teks terlebih dahulu.")


# ======= FOOTER =======
st.markdown("---")
st.caption("© 2025 Muhamad Rizal Rifaldi | Deteksi Hoax AI – Mode Gelap Elegan")
