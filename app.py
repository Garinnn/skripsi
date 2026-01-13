import streamlit as st
import joblib
import re
import string
import time

# ==============================================
# 🔹 KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# ==============================================
st.set_page_config(
    page_title="Deteksi Hoax Indonesia",
    layout="wide"
)

# ==============================================
# 🔹 FUNGSI MEMBERSIHKAN TEKS
# ==============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================================
# 🔹 LOAD MODEL DAN VECTORIZER (CACHE)
# ==============================================
@st.cache_resource
def load_models():
    try:
        model_berita = joblib.load("model_hoax_berita.pkl")
        vectorizer_berita = joblib.load("tfidf_vectorizer_berita.pkl")
        model_sosmed = joblib.load("model_hoax_sosmed.pkl")
        vectorizer_sosmed = joblib.load("tfidf_vectorizer_sosmed.pkl")
        return model_berita, vectorizer_berita, model_sosmed, vectorizer_sosmed
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan. Pastikan file .pkl ada di folder yang sama.")
        return None, None, None, None

# Load model
mnb_berita, vec_berita, mnb_sosmed, vec_sosmed = load_models()

# ==============================================
# 🔹 CSS CUSTOM
# ==============================================
st.markdown("""
<style>
body, .stApp {
    background-color: black;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.stMarkdown p {
    color: white !important;
    font-size: 1.1em;
}

.stTabs [role="tab"] {
    background-color: #f5f5f5;
    border-radius: 8px;
    color: #333;
    padding: 10px 20px;
    margin: 4px;
}

.stTabs [role="tab"][aria-selected="true"] {
    background-color: #007bff;
    color: white !important;
    font-weight: 700;
}

button[kind="primary"] {
    background-color: #007bff !important;
    color: white !important;
    border-radius: 6px;
    font-weight: 600;
}

textarea {
    background-color: #fafafa !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================
# 🔹 JUDUL
# ==============================================
st.title("Identifikasi Berita Hoax")
st.markdown(
    "Aplikasi ini membantu mengidentifikasi berita atau postingan sosial media "
    "apakah termasuk **REAL** atau **HOAX**."
)

# ==============================================
# 🔹 FUNGSI HASIL PREDIKSI
# ==============================================
def show_prediction_results(prob, pred, input_text):
    label_text = "REAL" if pred == 0 else "HOAX"
    emoji = "✅" if pred == 0 else "❌"

    st.markdown("---")
    st.subheader("📊 Hasil Deteksi")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Prediksi Model", f"{label_text} {emoji}")
    with col2:
        confidence = prob[1] if pred == 1 else prob[0]
        st.progress(confidence, text=f"Keyakinan: {confidence:.2f}")

    with st.expander("📋 Detail probabilitas"):
        st.write(f"REAL : {prob[0]*100:.2f}%")
        st.write(f"HOAX : {prob[1]*100:.2f}%")

    st.info(f"Teks dianalisis:\n\n> {input_text}")

# ==============================================
# 🔹 TAB
# ==============================================
tab1, tab2 = st.tabs(["Analisis Berita", "Analisis Sosial Media"])

with tab1:
    teks_berita = st.text_area("Masukkan teks berita", height=180)
    if st.button("🚀 Deteksi Berita", type="primary", use_container_width=True):
        if teks_berita.strip():
            with st.spinner("Menganalisis..."):
                time.sleep(1)
                fitur = vec_berita.transform([clean_text(teks_berita)])
                prob = mnb_berita.predict_proba(fitur)[0]
                pred = mnb_berita.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_berita)
        else:
            st.warning("⚠️ Masukkan teks berita!")

with tab2:
    teks_sosmed = st.text_area("Masukkan teks sosial media", height=180)
    if st.button("🚀 Deteksi Sosial Media", type="primary", use_container_width=True):
        if teks_sosmed.strip():
            with st.spinner("Menganalisis..."):
                time.sleep(1)
                fitur = vec_sosmed.transform([clean_text(teks_sosmed)])
                prob = mnb_sosmed.predict_proba(fitur)[0]
                pred = mnb_sosmed.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_sosmed)
        else:
            st.warning("⚠️ Masukkan teks sosial media!")

st.markdown("---")
st.caption("© 2025 Muhamad Rizal Rifaldi | Deteksi Hoax")
