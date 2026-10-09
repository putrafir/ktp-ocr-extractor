import streamlit as st
import json
import time
import io
import importlib
from pathlib import Path
from PIL import Image, ImageOps

# Enable HEIC / HEIF support for Apple iPhone photos
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import src.pipeline
importlib.reload(src.pipeline)
from src.pipeline import KTPExtractionPipeline
from src.config import Config

st.set_page_config(
    page_title="KTP Information Extractor & Validator",
    page_icon="🪪",
    layout="wide"
)

st.title("🪪 KTP Information Extractor & Validator")
st.markdown("""
Prototype ekstraksi informasi KTP lokal berbasis deep learning & penalaran spasial.
Dilengkapi **Two-Tier Document Validator** untuk memverifikasi apakah berkas yang diunggah adalah KTP sah atau bukan.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.info("🚀 **OCR Engine:** PaddleOCR (RapidOCR ONNX Runtime)")
    
    max_dim = st.number_input("Max Dimension (px)", value=1200, step=100)
    strict_validation = st.checkbox(
        "Strict KTP Validation",
        value=True,
        help="Tolak dokumen jika terdeteksi bukan KTP Indonesia yang sah (mencegah data halusinasi)"
    )
    
    if st.button("🔄 Bersihkan Cache Pipeline", help="Muat ulang model OCR dan pipeline ke versi terbaru"):
        st.cache_resource.clear()
        st.success("Cache berhasil dibersihkan!")
        st.rerun()

    st.divider()
    st.markdown("### 📌 Spesifikasi Arsitektur")
    st.markdown("""
    - **Model Spec:** Local Model (Zero Cloud SaaS)
    - **Validation:** Two-Tier Hybrid (Visual Sanity + Semantic Layout)
    - **OCR Engine:** PaddleOCR (DBNet + SVTR deep learning via ONNX Runtime)
    - **Preprocessing:** Auto-Deskew + HEIC iPhone support
    - **Latency Focus:** Yes (~500 - 800 ms)
    """)

# Pipeline instance cached with module reload & cache bust
@st.cache_resource(show_spinner=False)
def get_pipeline(max_d: int, strict_mode: bool, _v: int = 5):
    cfg = Config(MAX_IMAGE_WIDTH=max_d, OCR_DET_UNCLIP_RATIO=2.2)
    return KTPExtractionPipeline(config=cfg, strict_validation=strict_mode)

pipeline = get_pipeline(max_dim, strict_validation, _v=5)

# Sample file selector
samples_dir = Path("data/samples")
sample_files = (
    list(samples_dir.glob("*.jpg")) +
    list(samples_dir.glob("*.jpeg")) +
    list(samples_dir.glob("*.png")) +
    list(samples_dir.glob("*.heic")) +
    list(samples_dir.glob("*.HEIC"))
)
sample_names = ["-- Pilih dari sampel data --"] + [f.name for f in sample_files]

col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    uploaded_file = st.file_uploader(
        "Unggah Gambar KTP (JPG, PNG, HEIC iPhone, WEBP):",
        type=["jpg", "jpeg", "png", "heic", "heif", "webp"],
        help="Unggah foto atau scan KTP untuk diekstrak (Mendukung foto langsung dari iPhone/HEIC)"
    )

with col_input2:
    selected_sample = st.selectbox(
        "Atau pilih sampel yang sudah ada:",
        sample_names,
        index=0
    )

target_image_bytes = None
image_source_label = ""

if uploaded_file is not None:
    target_image_bytes = uploaded_file.read()
    image_source_label = uploaded_file.name
elif selected_sample != "-- Pilih dari sampel data --":
    sample_path = samples_dir / selected_sample
    if sample_path.exists():
        target_image_bytes = sample_path.read_bytes()
        image_source_label = selected_sample

if target_image_bytes:
    col_img, col_res = st.columns([1, 1], gap="medium")

    extracted_data = None
    preprocessed_bgr = None
    result = None

    with col_res:
        st.subheader("⚡ Proses Ekstraksi & Validasi")
        with st.spinner("Mengekstrak dan memvalidasi dokumen dengan PaddleOCR..."):
            try:
                try:
                    result = pipeline.process(target_image_bytes, strict_validation=strict_validation)
                except TypeError:
                    setattr(pipeline, "strict_validation", strict_validation)
                    result = pipeline.process(target_image_bytes)

                status = result.get("status", "success")
                val = result.get("validation", {})
                latency_ms = result.get("latency_ms", 0)
                items_count = result.get("text_items_count", 0)
                preprocessed_bgr = result.get("preprocessed_img")
                extracted_data = result.get("data")
                
                # Metrics banner
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric(label="⏱️ Total Latensi", value=f"{latency_ms} ms")
                with mcol2:
                    st.metric(label="🚀 Engine", value="PaddleOCR")
                with mcol3:
                    st.metric(label="🔍 Text Elements", value=f"{items_count} items")

                if status == "rejected":
                    st.error(f"❌ Dokumen Ditolak: Berkas BUKAN KTP Republik Indonesia yang valid (Skor: {val.get('confidence_score', 0)*100:.1f}%)")
                    with st.expander("🔍 Rincian Analisis Validasi", expanded=True):
                        st.write(f"**Skor Visual:** {val.get('visual_score', 0)*100:.1f}% | **Skor Struktur Semantik:** {val.get('semantic_score', 0)*100:.1f}%")
                        st.markdown("**Alasan Penolakan:**")
                        for r in val.get("rejection_reasons", []):
                            st.markdown(f"- ❌ {r}")
                else:
                    conf_pct = val.get('confidence_score', 0) * 100
                    st.success(f"✅ Dokumen KTP Terverifikasi (Keyakinan: {conf_pct:.1f}%) | Engine: PaddleOCR")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat ekstraksi: {e}")
                extracted_data = None
                preprocessed_bgr = None

    with col_img:
        st.subheader("🖼️ Preview Gambar")
        if preprocessed_bgr is not None:
            import cv2
            rgb_display = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb_display, caption=f"Input: {image_source_label}", width="stretch")
        else:
            try:
                pil_view = Image.open(io.BytesIO(target_image_bytes))
                pil_view = ImageOps.exif_transpose(pil_view).convert("RGB")
                st.image(pil_view, caption=f"Input: {image_source_label}", width="stretch")
            except Exception:
                st.image(target_image_bytes, caption=f"Input: {image_source_label}", width="stretch")

    if result:
        st.divider()
        if extracted_data:
            st.subheader("📋 Hasil Ekstraksi Key-Value Pair")
            tab_table, tab_json, tab_raw, tab_diag = st.tabs(["📊 Tabel Data", "💻 Raw JSON", "📝 Teks OCR Mentah", "🛡️ Rincian Validasi"])

            with tab_table:
                formatted_data = []
                field_labels = {
                    "nik": "NIK",
                    "nama": "Nama Lengkap",
                    "tempat_tgl_lahir": "Tempat / Tgl Lahir",
                    "jenis_kelamin": "Jenis Kelamin",
                    "gol_darah": "Golongan Darah",
                    "alamat": "Alamat",
                    "rt_rw": "RT / RW",
                    "kel_desa": "Kelurahan / Desa",
                    "kecamatan": "Kecamatan",
                    "agama": "Agama",
                    "status_perkawinan": "Status Perkawinan",
                    "pekerjaan": "Pekerjaan",
                    "kewarganegaraan": "Kewarganegaraan",
                    "berlaku_hingga": "Berlaku Hingga"
                }
                
                for key, label in field_labels.items():
                    val_field = extracted_data.get(key)
                    formatted_data.append({
                        "Field": label,
                        "Nilai": val_field if val_field else "-"
                    })
                
                st.table(formatted_data)

            with tab_json:
                json_str = json.dumps(extracted_data, ensure_ascii=False, indent=2)
                st.code(json_str, language="json")
                st.download_button(
                    label="📥 Unduh Hasil JSON",
                    data=json_str,
                    file_name="ktp_extracted.json",
                    mime="application/json"
                )

            with tab_raw:
                st.text_area("OCR Raw Text Layout:", value=result.get("raw_text", ""), height=200)

            with tab_diag:
                st.json(result.get("validation", {}))
        else:
            # Document rejected
            tab_raw, tab_diag = st.tabs(["📝 Teks OCR Mentah", "🛡️ Rincian Validasi"])
            with tab_raw:
                st.text_area("OCR Raw Text Layout:", value=result.get("raw_text", ""), height=200)
            with tab_diag:
                st.json(result.get("validation", {}))

else:
    st.info("👋 Silakan unggah gambar KTP di atas atau pilih sampel dari folder `data/samples/` untuk memulai.")
