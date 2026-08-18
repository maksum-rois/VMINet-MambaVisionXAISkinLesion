import os
import json
import time
import cv2
import torch
import timm
import hashlib
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. KONFIGURASI Halaman & PATH RELATIF
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# URL GitHub Release & SHA256 Hash
MAMBA_URL = "https://github.com/maksum-rois/VMINet-MambaVisionXAISkinLesion/releases/download/v1/mambavision_best.pt"
MAMBA_SHA256 = "2363e55448f8d8a1a998c9db95a21bf538a7dc7d7522004e7425ff9ef6c604aa"

VMINET_URL = "https://github.com/maksum-rois/VMINet-MambaVisionXAISkinLesion/releases/download/v1/vminet_best.pt"

st.set_page_config(
    page_title="Skin Lesion XAI Framework",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Framework Klasifikasi Lesi Kulit Berbasis Mamba & XAI")
st.caption("Integrasi MambaVision/VMINet, Preprocessing DullRazor+CLAHE, dan Visual XAI (Grad-CAM++, Score-CAM)")

# ==========================================
# 2. VERIFIKASI HASH & DOWNLOAD OTOMATIS
# ==========================================
def verify_sha256(file_path, expected_hash):
    """Memeriksa integritas file berdasarkan nilai SHA256 checksum."""
    if not os.path.exists(file_path):
        return False
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_hash.lower()

def download_model_with_integrity_check(file_path, url, expected_sha256=None):
    """Mengunduh file bobot model jika belum ada atau nilai Hash SHA256 tidak cocok."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    filename = os.path.basename(file_path)
    
    # Cek ketersediaan dan validasi SHA256
    if expected_sha256 and verify_sha256(file_path, expected_sha256):
        return  # File sudah ada dan valid
    
    with st.spinner(f"📦 Mengunduh bobot model {filename} (~181 MB)... Mohon tunggu sebentar."):
        try:
            urllib.request.urlretrieve(url, file_path)
            if expected_sha256 and not verify_sha256(file_path, expected_sha256):
                st.error(f"❌ Korupsi file terdeteksi pada {filename}. Nilai SHA256 tidak cocok.")
                os.remove(file_path)
        except Exception as e:
            st.error(f"❌ Gagal mengunduh {filename} dari GitHub Releases. Error: {e}")

# ==========================================
# 3. PEMUATAN CLASS MAPPING
# ==========================================
@st.cache_data
def load_class_mapping(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                raw_mapping = json.load(f)
            first_key = list(raw_mapping.keys())[0]
            if first_key.isdigit():
                return {int(k): str(v) for k, v in raw_mapping.items()}
            else:
                return {int(v): str(k) for k, v in raw_mapping.items()}
        except Exception:
            pass
    return {
        0: "nv (Melanocytic nevi)",
        1: "mel (Melanoma)",
        2: "bkl (Benign keratosis-like)",
        3: "bcc (Basal cell carcinoma)",
        4: "akiec (Actinic keratoses)",
        5: "vasc (Vascular lesions)",
        6: "df (Dermatofibroma)"
    }

CLASS_NAMES = load_class_mapping(os.path.join(MODEL_DIR, "class_mapping.json"))

# ==========================================
# 4. PREPROCESSING & HELPER FUNCTIONS
# ==========================================
def apply_dullrazor(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean = cv2.inpaint(img_rgb, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    return clean, mask

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

def get_last_spatial_layer(model):
    spatial_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or "stage" in name.lower() or "block" in name.lower():
            spatial_layers.append(module)
    return spatial_layers[-1] if spatial_layers else None

# ==========================================
# 5. MEMUAT MODEL (CACHED WITH VERIFICATION)
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(CLASS_NAMES)
    
    mamba_path = os.path.join(MODEL_DIR, "mambavision_best.pt")
    vminet_path = os.path.join(MODEL_DIR, "vminet_best.pt")
    
    # Auto-download dengan verifikasi SHA-256
    download_model_with_integrity_check(mamba_path, MAMBA_URL, MAMBA_SHA256)
    download_model_with_integrity_check(vminet_path, VMINET_URL)

    # 1. MambaVision Model
    mamba = timm.create_model('mambaout_small.in1k', pretrained=False, num_classes=num_classes)
    if os.path.exists(mamba_path):
        mamba.load_state_dict(torch.load(mamba_path, map_location=device))
    mamba.to(device).eval()

    # 2. VMINet Model
    vminet = timm.create_model('efficientformerv2_s0', pretrained=False, num_classes=num_classes)
    if os.path.exists(vminet_path):
        vminet.load_state_dict(torch.load(vminet_path, map_location=device))
    vminet.to(device).eval()

    return mamba, vminet, device

model_mamba, model_vminet, device = load_models()

# ==========================================
# 6. STRUKTUR 5 TAB PENELITIAN
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview", 
    "🧹 Preprocessing Engine", 
    "⚡ Model Efficiency (RO1)", 
    "🔬 Interpretability XAI (RO2)", 
    "🩺 Live Inference & Testing"
])

# ------------------------------------------
# TAB 1: DATASET OVERVIEW
# ------------------------------------------
with tab1:
    st.header("Metodologi Data & Stratifikasi Pasien")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Karakteristik Dataset")
        st.markdown(f"""
        * **HAM10000 Dataset**: Total 10,015 citra dermoskopi.
        * **Patient-Stratified Split**: 70% Train, 15% Validation, 15% Test.
        * **Pencegahan Leakage**: Pemisahan dikunci berbasis `lesion_id` / `patient_id`.
        * **Jumlah Kelas Terdaftar**: **{len(CLASS_NAMES)}** Kelas Diagnostik.
        """)
    with col2:
        st.subheader("Distribusi Kelas Diagnostik")
        chart_data = pd.DataFrame(
            {"Jumlah Sampel": [6705, 1113, 1099, 514, 327, 142, 115]}, 
            index=["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
        )
        st.bar_chart(chart_data)

# ------------------------------------------
# TAB 2: PREPROCESSING ENGINE
# ------------------------------------------
with tab2:
    st.header("Pipeline Preprocessing: DullRazor + CLAHE")
    st.markdown("Algoritma otomatis untuk menghilangkan artefak rambut dan menajamkan kontras batas lesi.")
    
    test_f = st.file_uploader("Unggah Sampel Gambar untuk Menguji Pipeline:", type=["jpg", "png", "jpeg"], key="p_up")
    if test_f:
        file_bytes = np.asarray(bytearray(test_f.read()), dtype=np.uint8)
        raw_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
        
        clean_rgb, mask = apply_dullrazor(raw_rgb)
        final_rgb = apply_clahe(clean_rgb)
        
        ca, cb, cc, cd = st.columns(4)
        ca.image(raw_rgb, caption="1. Citra Mentah (Raw)", use_container_width=True)
        cb.image(mask, caption="2. Masker Rambut (DullRazor)", use_container_width=True)
        cc.image(clean_rgb, caption="3. Inpainting (Cleaned)", use_container_width=True)
        cd.image(final_rgb, caption="4. CLAHE (Final Input)", use_container_width=True)

# ------------------------------------------
# TAB 3: BENCHMARK METRICS (RO1)
# ------------------------------------------
with tab3:
    st.header("Hasil Tolok Ukur Efisiensi Komputasi (RO1)")
    csv_path = os.path.join(DATA_DIR, "ro1_benchmark_metrics.csv")
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    else:
        st.info("File metrik `ro1_benchmark_metrics.csv` tidak ditemukan di folder `data/`.")

# ------------------------------------------
# TAB 4: INTERPRETABILITY XAI (RO2)
# ------------------------------------------
with tab4:
    st.header("Framework Visual XAI (RO2)")
    st.markdown("""
    * **Grad-CAM++**: Pemetaan atribusi fitur visual berbasis turunan/gradien tingkat tinggi.
    * **Score-CAM**: Pemetaan fitur independen-gradien menggunakan skor aktivasi langsung.
    * **Visualisasi Kontras**: Area merah menunjukkan wilayah utama yang memicu keputusan model.
    """)
    st.info("💡 Buka **Tab 5** untuk menguji visualisasi XAI secara langsung dengan citra medis pilihan Anda.")

# ------------------------------------------
# TAB 5: LIVE INFERENCE & TESTING
# ------------------------------------------
with tab5:
    st.header("Modul Pengujian Live Diagnosis & XAI")
    
    col_sel, col_up = st.columns([1, 2])
    with col_sel:
        m_choice = st.selectbox("Pilih Model Arsitektur:", ["MambaVision", "VMINet"])
    with col_up:
        inf_f = st.file_uploader("Unggah Citra Dermoskopi Pasien:", type=["jpg", "png", "jpeg"], key="l_up")
        
    if inf_f:
        act_model = model_mamba if m_choice == "MambaVision" else model_vminet
        
        file_bytes = np.asarray(bytearray(inf_f.read()), dtype=np.uint8)
        raw_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        clean_rgb, _ = apply_dullrazor(raw_rgb)
        final_rgb = apply_clahe(clean_rgb)
        
        # Tensor Transformation
        pil_img = Image.fromarray(final_rgb).resize((224, 224))
        img_arr = np.array(pil_img) / 255.0
        tensor_in = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor_norm = norm(tensor_in).to(device)
        
        # Inferensi
        start = time.time()
        with torch.no_grad():
            outputs = act_model(tensor_norm)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        lat = (time.time() - start) * 1000
        
        p_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES.get(p_idx, f"Class {p_idx}")
        
        st.success(f"🔍 Prediksi: **{pred_label}** | Keyakinan: **{probs[p_idx]*100:.2f}%** | Latensi CPU: **{lat:.1f} ms**")
        
        # Generate Map XAI
        target_layer = get_last_spatial_layer(act_model)
        if target_layer:
            st.subheader("Peta Penjelasan Visual (XAI Heatmap)")
            
            try:
                gpp = GradCAMPlusPlus(model=act_model, target_layers=[target_layer])
                vis_gpp = show_cam_on_image(np.float32(pil_img)/255.0, gpp(input_tensor=tensor_norm)[0, :], use_rgb=True)
                
                sc = ScoreCAM(model=act_model, target_layers=[target_layer])
                vis_sc = show_cam_on_image(np.float32(pil_img)/255.0, sc(input_tensor=tensor_norm)[0, :], use_rgb=True)
                
                xa, xb, xc = st.columns(3)
                xa.image(final_rgb, caption="Hasil Preprocessing", use_container_width=True)
                xb.image(vis_gpp, caption="Grad-CAM++ Heatmap", use_container_width=True)
                xc.image(vis_sc, caption="Score-CAM Heatmap", use_container_width=True)
            except Exception as e:
                st.warning(f"Gagal memuat peta XAI: {str(e)}")
