import streamlit as st
import json
from pathlib import Path
from PIL import Image, ImageOps
import io

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import KTPExtractionPipeline
from src.config import Config

# Page setup
st.set_page_config(
    page_title="KTP Extractor Prototype",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🪪 KTP Information Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Local Model Prototype: OCR + Spatial Layout Extractor (PaddleOCR & LiteParse)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi Engine")
    ocr_engine = st.selectbox(
        "Pilih OCR Engine:",
        ["paddle", "liteparse"],
        format_func=lambda x: "PaddleOCR (Rekomendasi: Akurasi Tinggi)" if x == "paddle" else "LiteParse (Eksperimen Layout)",
        index=0
    )
    max_dim = st.number_input("Max Dimension (px)", value=1200, step=100)
    
    st.divider()
    st.markdown("### 📌 Spesifikasi Arsitektur")
    st.markdown("""
    - **Model Spec:** Local Model (Zero Cloud SaaS)
    - **Engines:**
      - **PaddleOCR:** DBNet + SVTR deep learning (OnnxRuntime)
      - **LiteParse:** Spatial grid projection (PDFium/Rust)
    - **Preprocessing:** Auto-Deskew + HEIC iPhone support
    - **Latency Focus:** Yes (~500 - 900 ms)
    - **YOLO / Fine-tuning:** None (Rule & Geometry based)
    """)

# Pipeline instance cached
@st.cache_resource
def get_pipeline(engine_name: str, max_d: int):
    cfg = Config(MAX_IMAGE_WIDTH=max_d)
    return KTPExtractionPipeline(config=cfg, engine=engine_name)

pipeline = get_pipeline(ocr_engine, max_dim)

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

    with col_res:
        st.subheader("⚡ Proses Ekstraksi")
        with st.spinner(f"Mengekstrak data KTP dengan {ocr_engine.upper()}..."):
            try:
                result = pipeline.process(target_image_bytes)
                extracted_data = result["data"]
                latency_ms = result["latency_ms"]
                items_count = result["text_items_count"]
                preprocessed_bgr = result.get("preprocessed_img")
                
                # Metrics banner
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric(label="⏱️ Total Latensi", value=f"{latency_ms} ms")
                with mcol2:
                    st.metric(label="🚀 Engine", value=result["engine"].upper())
                with mcol3:
                    st.metric(label="🔍 Text Elements", value=f"{items_count} items")

                st.success(f"Ekstraksi Berhasil menggunakan {result['engine'].upper()}!")

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

    if extracted_data:
        st.divider()
        st.subheader("📋 Hasil Ekstraksi Key-Value Pair")
        
        tab_table, tab_json, tab_raw = st.tabs(["📊 Tabel Data", "💻 Raw JSON", "📝 Teks OCR Mentah"])

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
                val = extracted_data.get(key)
                formatted_data.append({
                    "Field": label,
                    "Nilai": val if val else "-"
                })
            
            st.table(formatted_data)

        with tab_json:
            json_str = json.dumps(extracted_data, ensure_ascii=False, indent=2)
            st.code(json_str, language="json")
            st.download_button(
                label="📥 Unduh Hasil JSON",
                data=json_str,
                file_name=f"ktp_{ocr_engine}.json",
                mime="application/json"
            )

        with tab_raw:
            st.text_area("OCR Raw Text Layout:", value=result.get("raw_text", ""), height=200)

else:
    st.info("👋 Silakan unggah gambar KTP di atas atau taruh file KTP di folder `data/samples/` untuk memulai.")
