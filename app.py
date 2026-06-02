
import streamlit as st
import pickle
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Kesiapan Karir Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #4a00e0 0%, #6a11cb 50%, #2575fc 100%);
    padding: 2rem 2.5rem; border-radius: 18px; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(74,0,224,0.3);
    text-align: center;
}
.main-header h1 { font-size: 2rem; font-weight: 800; margin: 0; }
.main-header p  { font-size: 0.95rem; opacity: 0.9; margin: 0.4rem 0 0; }
.verdict-box {
    border-radius: 14px; padding: 1.5rem; text-align: center;
    margin-top: 1rem; font-weight: 700; font-size: 1.6rem;
}
.sangat-siap     { background: linear-gradient(135deg, #d4edda, #a8dfc0); border: 2px solid #28a745; color: #155724; }
.siap-industri   { background: linear-gradient(135deg, #cce5ff, #99c8ff); border: 2px solid #007bff; color: #004085; }
.hampir-siap     { background: linear-gradient(135deg, #fff3cd, #ffd77d); border: 2px solid #ffc107; color: #856404; }
.perlu-persiapan { background: linear-gradient(135deg, #f8d7da, #f5a5ab); border: 2px solid #dc3545; color: #721c24; }
.metric-card {
    background: #f8f9fa; border-radius: 12px; padding: 1rem;
    text-align: center; border: 1px solid #e9ecef;
    margin-bottom: 0.5rem;
}
.metric-card h3 { font-size: 0.8rem; color: #6c757d; margin: 0; }
.metric-card p  { font-size: 1.4rem; font-weight: 700; color: #212529; margin: 0.2rem 0 0; }
.sidebar-info {
    background: #f0f4ff; border: 1px solid #c5d3ff;
    padding: 1rem; border-radius: 12px; font-size: 0.83rem; line-height: 1.7;
}
.stButton > button {
    width: 100%; background: linear-gradient(135deg, #4a00e0, #2575fc);
    color: white; border: none; border-radius: 10px;
    padding: 0.75rem; font-weight: 700; font-size: 1rem;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(path='model_kesiapan_karir.pkl'):
    if not os.path.exists(path):
        return None, f'File model tidak ditemukan: {path}'
    try:
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        return bundle, None
    except Exception as e:
        return None, str(e)

def get_verdict_css(verdict):
    css_map = {
        'Sangat Siap'        : ('sangat-siap',     '🏆'),
        'Siap Masuk Industri': ('siap-industri',   '✅'),
        'Hampir Siap'        : ('hampir-siap',     '⚡'),
        'Perlu Persiapan'    : ('perlu-persiapan', '📚')
    }
    return css_map.get(verdict, ('siap-industri', '❓'))

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('## ⚙️ Pengaturan')
    mode = st.radio(
        'Sumber Model:',
        ['🔍 Gunakan model lokal', '📂 Upload model (.pkl)'],
        index=0
    )
    bundle = None
    if mode == '🔍 Gunakan model lokal':
        bundle, err = load_model()
        if bundle:
            st.success('✅ Model berhasil dimuat!')
        else:
            st.error(f'❌ {err}')
    else:
        up = st.file_uploader('Upload model_kesiapan_karir.pkl', type=['pkl'])
        if up:
            try:
                bundle = pickle.load(up)
                st.success('✅ Model berhasil dimuat!')
            except Exception as e:
                st.error(f'❌ {e}')
    st.markdown('---')
    st.markdown("""
    <div class="sidebar-info">
    <b>🎯 Kategori Verdict:</b><br>
    🏆 <b>Sangat Siap</b> — Siap kerja penuh<br>
    ✅ <b>Siap Masuk Industri</b> — Siap dengan minor gap<br>
    ⚡ <b>Hampir Siap</b> — Perlu persiapan tambahan<br>
    📚 <b>Perlu Persiapan</b> — Masih butuh banyak persiapan<br><br>
    <b>🤖 Algoritma:</b> XGBoost<br>
    <b>📊 Fitur:</b> Skor asesmen + latar belakang akademik
    </div>
    """, unsafe_allow_html=True)
    tz = pytz.timezone('Asia/Jakarta')
    st.caption(f'🕒 {datetime.now(tz).strftime("%d %B %Y, %H:%M")} WIB')

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
  <h1>🎓 Sistem Klasifikasi Kesiapan Karir Mahasiswa</h1>
  <p>Prediksi tingkat kesiapan karir mahasiswa berbasis hasil asesmen skill &amp; latar belakang akademik menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.warning('⚠️ Model belum tersedia. Silakan load model dari sidebar.')
    st.stop()

model       = bundle['model']
preprocessor= bundle['preprocessor']
num_cols    = bundle['numeric_cols']
cat_cols    = bundle['categorical_cols']
label_decode= bundle['label_decode']
feat_names  = bundle['feature_names']

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    '✍️ Prediksi Manual',
    '📊 Prediksi Massal (CSV)',
    '📖 Panduan'
])

# =========================================================
# TAB 1 — PREDIKSI MANUAL
# =========================================================
with tab1:
    st.subheader('✍️ Input Data Mahasiswa untuk Prediksi')
    st.markdown('Isi formulir berikut sesuai data mahasiswa yang ingin diprediksi.')

    with st.form('form_prediksi'):
        st.markdown('#### 📋 Data Akademik')
        col1, col2, col3 = st.columns(3)
        with col1:
            semester = st.slider('Semester', 1, 14, 5)
            ipk      = st.slider('IPK', 0.0, 4.0, 3.2, 0.01)
        with col2:
            kesiapan_interview = st.slider('Kesiapan Interview (1-5)', 1, 5, 3)
            level_bi           = st.slider('Level Bahasa Inggris (1-5)', 1, 5, 3)
        with col3:
            minat_wirausaha  = st.slider('Minat Wirausaha (1-5)', 1, 5, 3)
            rencana_s2       = st.slider('Rencana S2/Sertifikasi (1-5)', 1, 5, 3)
            frekuensi_belajar= st.slider('Frekuensi Belajar (1-5)', 1, 5, 3)
            ekspektasi_gaji  = st.slider('Ekspektasi Gaji (1-5)', 1, 5, 3)

        st.markdown('#### 🎯 Skor Asesmen (M1–M9)')
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            m1 = st.slider('M1 Background', 0, 15, 8)
        with col_m2:
            m2 = st.slider('M2 Skills', 0, 15, 9)
        with col_m3:
            m3 = st.slider('M3 Industry', 0, 15, 8)
        with col_m4:
            m4 = st.slider('M4 Interest', 0, 15, 10)
        with col_m5:
            m5 = st.slider('M5 Compass', 0, 15, 9)
        col_m6, col_m7, col_m8, col_m9, col_ts = st.columns(5)
        with col_m6:
            m6 = st.slider('M6 Company', 0, 15, 9)
        with col_m7:
            m7 = st.slider('M7 Branding', 0, 15, 8)
        with col_m8:
            m8 = st.slider('M8 Ambisi', 0, 15, 8)
        with col_m9:
            m9 = st.slider('M9 Resiliensi', 0, 15, 8)

        st.markdown('#### 🌐 Preferensi & Karakteristik Karir')
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            jalur_karir = st.selectbox('Jalur Karir', ['expand', 'specialist', 'entrepreneur', 'academia'])
            bidang_minat= st.text_input('Bidang Minat', 'tech, data')
            work_values = st.text_input('Work Values', 'growth, wlb')
        with col_b:
            work_style  = st.selectbox('Work Style', ['big', 'small', 'flex'])
            cognitive   = st.selectbox('Cognitive Style', ['analytical', 'creative', 'social'])
            motivasi    = st.selectbox('Motivasi', ['growth', 'impact', 'prestige', 'salary'])
        with col_c:
            ukuran      = st.selectbox('Ukuran Perusahaan', ['big', 'startup', 'medium', 'flex'])
            kultur      = st.selectbox('Kultur Kerja', ['agile', 'result', 'structured', 'collaborative'])
            lokasi      = st.selectbox('Preferensi Lokasi', ['hybrid', 'remote', 'onsite'])

        submitted = st.form_submit_button('🔍 Prediksi Sekarang')

    if submitted:
        # Buat dict input
        input_dict = {
            'Semester': semester, 'IPK': ipk,
            'Skor M1 (Background)': m1, 'Skor M2 (Skills)': m2,
            'Skor M3 (Industry)': m3, 'Skor M4 (Interest)': m4,
            'Skor M5 (Compass)': m5, 'Skor M6 (Company)': m6,
            'Skor M7 (Branding)': m7, 'Skor M8 (Ambisi)': m8,
            'Skor M9 (Resiliensi)': m9,
            'Kesiapan Interview': kesiapan_interview,
            'Level Bahasa Inggris': level_bi,
            'Ekspektasi Gaji': ekspektasi_gaji,
            'Minat Wirausaha': minat_wirausaha,
            'Rencana S2/Sertifikasi': rencana_s2,
            'Frekuensi Belajar': frekuensi_belajar,
            'Jalur Karir': jalur_karir,
            'Bidang Minat': bidang_minat,
            'Work Values': work_values,
            'Work Style': work_style,
            'Cognitive Style': cognitive,
            'Motivasi': motivasi,
            'Ukuran Perusahaan': ukuran,
            'Kultur Kerja': kultur,
            'Preferensi Lokasi': lokasi
        }
        input_df = pd.DataFrame([input_dict])

        # Reorder kolom sesuai urutan training
        input_df = input_df.reindex(columns=feat_names, fill_value=0)

        with st.spinner('⏳ Memproses prediksi...'):
            X_prep   = preprocessor.transform(input_df)
            pred_idx = model.predict(X_prep)[0]
            verdict  = label_decode[pred_idx]

            probas = None
            if hasattr(model, 'predict_proba'):
                probas  = model.predict_proba(X_prep)[0]
                classes = [label_decode[i] for i in range(len(probas))]

        css_class, emoji = get_verdict_css(verdict)
        st.markdown(
            f"<div class='verdict-box {css_class}'>{emoji} {verdict}</div>",
            unsafe_allow_html=True
        )

        if probas is not None:
            st.markdown('#### 📊 Distribusi Probabilitas')
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                for cls, p in zip(classes, probas):
                    st.write(f'**{cls}** : {p*100:.2f}%')
                    st.progress(float(p))
            with col_p2:
                fig, ax = plt.subplots(figsize=(5, 3))
                bar_colors = ['#2ecc71','#3498db','#f39c12','#e74c3c']
                ax.barh(classes, [p*100 for p in probas],
                        color=bar_colors[:len(classes)], alpha=0.85)
                ax.set_xlabel('Probabilitas (%)')
                ax.set_title('Distribusi Probabilitas Prediksi', fontweight='bold')
                ax.set_xlim(0, 115)
                for i, (cls, p) in enumerate(zip(classes, probas)):
                    ax.text(p*100+1, i, f'{p*100:.1f}%', va='center', fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# =========================================================
# TAB 2 — PREDIKSI MASSAL
# =========================================================
with tab2:
    st.subheader('📊 Prediksi Massal dari File CSV/Excel')
    st.markdown('Upload file CSV atau Excel yang memiliki kolom sesuai fitur model.')
    st.info('📌 Pastikan file memiliki kolom yang sama dengan data training.')

    uploaded_file = st.file_uploader('Upload file data (CSV/Excel)', type=['csv','xlsx'])

    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)

            # Validasi kolom
            missing_cols = [
                col for col in feat_names
                if col not in df_batch.columns
            ]
            if missing_cols:
                st.error(
                    f'❌ Kolom berikut belum ada:\n\n'
                    f'{missing_cols}'
                )
                st.stop()

            # Reorder kolom
            X_batch = df_batch.reindex(columns=feat_names, fill_value=0)

            # Preprocessing
            X_batch_prep = preprocessor.transform(X_batch)

            # Prediksi
            preds = model.predict(X_batch_prep)
            df_batch['Prediksi Verdict'] = [label_decode[p] for p in preds]

            # Probabilitas
            if hasattr(model, 'predict_proba'):
                probas_all = model.predict_proba(X_batch_prep)
                for i, cls in enumerate(label_decode.values()):
                    df_batch[f'Prob_{cls}'] = (probas_all[:, i] * 100).round(2)

            # Tampilkan hasil
            st.success(f'✅ Selesai! {len(df_batch):,} data diprediksi.')
            st.dataframe(df_batch[['Prediksi Verdict'] +
                                   [c for c in df_batch.columns if 'Prob_' in c]].head(20),
                         use_container_width=True)

            # Distribusi hasil
            st.markdown('#### 📌 Ringkasan Distribusi Prediksi')
            dist = df_batch['Prediksi Verdict'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(dist.index, dist.values,
                    color=['#2ecc71','#3498db','#f39c12','#e74c3c'][:len(dist)],
                    alpha=0.85)
            ax2.set_title('Distribusi Hasil Prediksi', fontweight='bold')
            for i, v in enumerate(dist.values):
                ax2.text(i, v+0.3, str(v), ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # Download hasil
            st.download_button(
                '⬇️ Download Hasil Prediksi CSV',
                data=df_batch.to_csv(index=False, encoding='utf-8-sig'),
                file_name='hasil_prediksi_kesiapan_karir.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f'❌ Error: {e}')

# =========================================================
# TAB 3 — PANDUAN
# =========================================================
with tab3:
    st.subheader('📖 Panduan Penggunaan Sistem')
    st.markdown("""
    #### 📌 Cara Menggunakan Aplikasi

    **1. Prediksi Manual**
    - Isi formulir data mahasiswa pada tab **Prediksi Manual**.
    - Masukkan data akademik, skor asesmen M1–M9, dan preferensi karir.
    - Klik tombol **🔍 Prediksi Sekarang** untuk mendapatkan hasil prediksi.

    **2. Prediksi Massal**
    - Upload file CSV atau Excel yang berisi data banyak mahasiswa.
    - Pastikan kolom file sesuai dengan fitur training model.
    - Download hasil prediksi dalam format CSV.

    #### 🏷️ Kategori Verdict Kesiapan
    | Verdict | Deskripsi |
    |---------|----------|
    | 🏆 Sangat Siap | Mahasiswa sangat siap memasuki dunia kerja |
    | ✅ Siap Masuk Industri | Siap dengan gap minor yang dapat diatasi |
    | ⚡ Hampir Siap | Perlu persiapan tambahan di beberapa aspek |
    | 📚 Perlu Persiapan | Masih memerlukan banyak pengembangan diri |

    #### ⚠️ Catatan Penting
    - Prediksi didasarkan pada kombinasi skor asesmen, akademik, dan preferensi karir mahasiswa.
    - Model ini merupakan alat bantu dan bukan keputusan final.
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center;color:#999;font-size:0.8rem;padding:1rem;margin-top:2rem;border-top:1px solid #eee;">
  🎓 Sistem Klasifikasi Kesiapan Karir Mahasiswa — Kelompok 1 | Final Project
</div>
""", unsafe_allow_html=True)
