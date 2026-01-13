import streamlit as st
import joblib
import re
import string
import time

# ==============================================
# 🔹 FUNGSI MEMBERSIHKAN TEKS
# Membersihkan URL, tanda baca, angka, dan spasi ganda
# ==============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================
# 🔹 LOAD MODEL DAN VECTORIZER (CACHE RESOURCE)
# Menghindari loading ulang model setiap interaksi
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


# Muat model
mnb_berita, vec_berita, mnb_sosmed, vec_sosmed = load_models()


# ==============================================
# 🔹 KONFIGURASI HALAMAN UTAMA
# ==============================================
st.set_page_config(page_title="Deteksi Hoax Indonesia", layout="wide")

# ==============================================
# 🔹 CSS CUSTOM UNTUK GAYA PUTIH & TEKS HITAM
# ==============================================
st.markdown("""
<style>
/* ====== BODY UMUM ====== */
body, .stApp {
    background-color: black;           /* warna dasar putih */
    color: white;                      /* teks utama hitam */
    font-family: 'Segoe UI', sans-serif;
}

/* ====== HEADER & TITLE ====== */
h1, h2, h3, h4 {
    color: black;
    font-weight: 700;
}

/* ====== TAB STYLING ====== */
.stTabs [role="tablist"] {
    justify-content: center;
    border-bottom: 2px solid #ddd;
}
.stTabs [role="tab"] {
    background-color: #f5f5f5;         /* warna tab tidak aktif abu muda */
    border-radius: 8px;
    color: #333;                       /* teks tab hitam keabu-abuan */
    padding: 10px 20px;
    margin: 4px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [role="tab"]:hover {
    background-color: #e0e0e0;         /* efek hover tab */
}
.stTabs [role="tab"][aria-selected="true"] {
    background-color: #007bff;         /* warna tab aktif biru */
    color: white !important;           /* teks putih biar kontras */
    font-weight: 700;
}
.stMarkdown p {
    color: white !important;    /* ubah ke warna teks yang lo mau, contoh: abu tua */
    font-size: 1.1em;             /* biar sedikit lebih besar */
}

/* ====== TOMBOL ====== */
button[kind="primary"] {
    background-color: #007bff !important;  /* biru utama */
    color: white !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
button[kind="primary"]:hover {
    background-color: #0056b3 !important;  /* biru gelap saat hover */
}

/* ====== TEXT AREA ====== */
textarea {
    background-color: #fafafa !important;
    color: black !important;
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
    font-size: 1em;
}

/* ====== FOOTER ====== */
footer, .stMarkdown {
    color: #333;
}
</style>
""", unsafe_allow_html=True)


# ==============================================
# 🔹 JUDUL & PENJELASAN APLIKASI
# ==============================================
st.title("Identifikasi Berita Hoax")
st.markdown("""
Aplikasi ini membantu anda mengidentifikasi sebuah berita atau postingan sosial media termasuk kedalam kategori Real atau Hoax
""")


# ==============================================
# 🔹 FUNGSI UNTUK MENAMPILKAN HASIL PREDIKSI
# ==============================================
def show_prediction_results(prob, pred, input_text):
    label_text = "REAL" if pred == 0 else "HOAX"
    emoji = "✅" if pred == 0 else "❌"

    st.markdown("---")
    st.subheader("📊 Hasil Deteksi")

    # Tampilan hasil ringkas
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Prediksi Model", value=f"{label_text} {emoji}")
    with col2:
        st.markdown(f"**Probabilitas Keyakinan**")
        st.progress(prob[1] if pred == 1 else prob[0], 
                    text=f"Keyakinan: {prob[1] if pred == 1 else prob[0]:.2f}")

    # Detail tambahan
    with st.expander("📋 Lihat detail probabilitas"):
        st.write(f"**Probabilitas REAL:** {prob[0]*100:.2f}%")
        st.write(f"**Probabilitas HOAX:** {prob[1]*100:.2f}%")

    st.info(f"Teks yang dianalisis:\n\n> {input_text}")


# ==============================================
# 🔹 TAB 1 — Analisis Berita
# ==============================================
tab1, tab2 = st.tabs(["Analisis Berita", "Analisis Sosial Media"])

with tab1:
    st.subheader("Masukkan teks berita di bawah ini:")
    teks_berita = st.text_area("", height=180, placeholder="Salin dan tempel teks berita di sini...")

    if st.button("🚀 Deteksi Berita", type="primary", use_container_width=True):
        if teks_berita.strip():
            with st.spinner("Model sedang menganalisis..."):
                time.sleep(1)
                teks_bersih = clean_text(teks_berita)
                fitur = vec_berita.transform([teks_bersih])
                prob = mnb_berita.predict_proba(fitur)[0]
                pred = mnb_berita.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_berita)
        else:
            st.warning("⚠️ Masukkan teks berita terlebih dahulu!")


# ==============================================
# 🔹 TAB 2 — Analisis Sosial Media
# ==============================================
with tab2:
    st.subheader("Masukkan teks postingan media sosial di bawah ini:")
    teks_sosmed = st.text_area("", height=180, key="sosmed_input", 
                               placeholder="Salin dan tempel postingan sosmed di sini...")

    if st.button("🚀 Deteksi Sosial Media", type="primary", use_container_width=True, key="btn_sosmed"):
        if teks_sosmed.strip():
            with st.spinner("Model sedang menganalisis..."):
                time.sleep(1)
                teks_bersih = clean_text(teks_sosmed)
                fitur = vec_sosmed.transform([teks_bersih])
                prob = mnb_sosmed.predict_proba(fitur)[0]
                pred = mnb_sosmed.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_sosmed)
        else:
            st.warning("⚠️ Masukkan teks postingan terlebih dahulu!")


# ==============================================
# 🔹 FOOTER
# ==============================================
st.markdown("---")
st.caption("© 2025 Muhamad Rizal Rifaldi | Deteksi Hoax")