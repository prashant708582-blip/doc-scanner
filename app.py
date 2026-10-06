import streamlit as st
import os
import cv2
import numpy as np

from scanner import (
    crop_document, enhance_for_ocr, auto_rotate,
    image_to_text, detect_tables,
    text_to_pdf, multi_page_pdf, images_to_pdf
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="DocScan Pro — Hindi + English OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

WORK_DIR = "work"
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        color: white;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #64748b;
        line-height: 1.5;
    }
    
    /* Upload area */
    .upload-area {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0 2rem 0;
    }
    
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
        color: white;
    }
    
    /* Result section */
    .result-header {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] h2 {
        color: #667eea;
        font-size: 1.1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        border-radius: 10px;
    }
    
    /* Info message */
    .stInfo {
        border-left: 5px solid #667eea;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HERO HEADER ----------
st.markdown("""
<div class="hero-header">
    <h1>📄 DocScan Pro</h1>
    <p>AI-powered document scanner with Hindi + English OCR</p>
    <div class="hero-badge">✨ Auto-crop • 🔍 Smart OCR • 📑 PDF Export</div>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("")

    lang_choice = st.selectbox(
        "🌐 OCR Language",
        ["hin+eng (Hindi + English)", "eng (English)", "hin (Hindi)"]
    )
    lang_code = {
        "hin+eng (Hindi + English)": "hin+eng",
        "eng (English)": "eng",
        "hin (Hindi)": "hin"
    }[lang_choice]

    psm_choice = st.selectbox(
        "🎯 OCR Mode",
        [
            (3, "Auto (mixed)"),
            (6, "Single block"),
            (4, "Multi-column"),
            (11, "Sparse text"),
            (12, "Sparse + OSD")
        ],
        format_func=lambda x: x[1]
    )
    psm_val = psm_choice[0]

    enhance_mode = st.selectbox(
        "🎨 Image Enhancement",
        ["auto", "bw", "grayscale", "color"],
        index=0
    )

    st.markdown("---")
    st.markdown("## 🔧 Advanced")
    
    handwriting = st.checkbox("✍️ Handwriting mode", value=False, help="Best-effort recognition for handwritten text")
    auto_rot = st.checkbox("🔄 Auto-rotate tilted pages", value=True)

    st.markdown("---")
    st.markdown("## 📑 Output Format")
    output_type = st.radio(
        "Choose",
        ["Both", "Text PDF (searchable)", "Image PDF (scan)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: #f0f4ff; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
        <div style="font-size: 0.85rem; color: #4c51bf; font-weight: 600; margin-bottom: 0.5rem;">💡 Pro Tip</div>
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
            For best results:<br>
            • 300 DPI scan<br>
            • Straight angle<br>
            • Good lighting
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- UPLOAD SECTION ----------
st.markdown("### 📤 Upload Your Document")
st.markdown("")

uploaded_files = st.file_uploader(
    "Choose files",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ---------- MAIN LOGIC ----------
if uploaded_files:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #d1fae5 0%, #a7f3d0 100%); 
                padding: 1rem 1.5rem; border-radius: 12px; 
                border-left: 5px solid #10b981; margin: 1rem 0;">
        <span style="color: #065f46; font-weight: 600;">
            ✅ {len(uploaded_files)} file(s) uploaded — processing started...
        </span>
    </div>
    """, unsafe_allow_html=True)

    all_texts = []
    all_cropped_paths = []
    all_enhanced_paths = []

    progress = st.progress(0)
    status = st.empty()

    for idx, uploaded in enumerate(uploaded_files):
        status.markdown(f"""
        <div style="background: #fef3c7; padding: 0.8rem 1.2rem; 
                    border-radius: 10px; border-left: 4px solid #f59e0b; 
                    margin: 0.5rem 0;">
            <span style="color: #92400e; font-weight: 600;">
                🔄 Processing: {uploaded.name} ({idx+1}/{len(uploaded_files)})
            </span>
        </div>
        """, unsafe_allow_html=True)
        progress.progress(idx / len(uploaded_files))

        orig_path = f"{WORK_DIR}/orig_{idx}.jpg"
        with open(orig_path, "wb") as f:
            f.write(uploaded.getbuffer())

        cropped_path = f"{WORK_DIR}/cropped_{idx}.jpg"
        try:
            crop_document(orig_path, cropped_path)
        except Exception as e:
            st.warning(f"Page {idx+1}: Crop fail — original use kar rahe hain")
            cropped_path = orig_path

        if auto_rot:
            cropped_img = cv2.imread(cropped_path)
            if cropped_img is not None:
                rotated = auto_rotate(cropped_img)
                rotated_path = f"{WORK_DIR}/rotated_{idx}.jpg"
                cv2.imwrite(rotated_path, rotated)
            else:
                rotated_path = cropped_path
        else:
            rotated_path = cropped_path

        enhanced = enhance_for_ocr(rotated_path, mode=enhance_mode)
        enhanced_path = f"{WORK_DIR}/enhanced_{idx}.jpg"
        cv2.imwrite(enhanced_path, enhanced)

        try:
            text = image_to_text(
                enhanced_path,
                lang=lang_code,
                psm=psm_val,
                handwriting=handwriting
            )
        except Exception as e:
            st.error(f"Page {idx+1}: OCR fail — {e}")
            text = ""

        all_texts.append(text)
        all_cropped_paths.append(cropped_path)
        all_enhanced_paths.append(enhanced_path)

    progress.progress(1.0)
    status.markdown("""
    <div style="background: linear-gradient(90deg, #d1fae5 0%, #a7f3d0 100%); 
                padding: 1rem 1.5rem; border-radius: 12px; 
                border-left: 5px solid #10b981; margin: 1rem 0;">
        <span style="color: #065f46; font-weight: 600;">
            ✨ All pages processed successfully!
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- STATS ----------
    st.markdown("### 📊 Summary")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    total_chars = sum(len(t) for t in all_texts)
    total_words = sum(len(t.split()) for t in all_texts)

    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{len(uploaded_files)}</p>
            <p class="stat-label">Pages</p>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{total_words}</p>
            <p class="stat-label">Words</p>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{total_chars}</p>
            <p class="stat-label">Characters</p>
        </div>
        """, unsafe_allow_html=True)

    with col_s4:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{len(all_texts)}</p>
            <p class="stat-label">OCR Done</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ---------- TABS ----------
    tab1, tab2, tab3 = st.tabs(["📄 Results", "📝 Text", "⬇️ Download"])

    # TAB 1: RESULTS
    with tab1:
        for idx, uploaded in enumerate(uploaded_files):
            with st.expander(f"📄 Page {idx+1} — {uploaded.name}", expanded=(idx == 0)):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**📷 Original**")
                    st.image(uploaded, use_column_width=True)

                with col2:
                    st.markdown("**✂️ Auto-Cropped**")
                    st.image(all_cropped_paths[idx], use_column_width=True)

                with col3:
                    st.markdown("**✨ Enhanced**")
                    st.image(all_enhanced_paths[idx], use_column_width=True)

                tables = detect_tables(all_enhanced_paths[idx])
                if tables:
                    st.info(f"📊 {len(tables)} table(s) detected on this page")

    # TAB 2: TEXT
    with tab2:
        for idx, uploaded in enumerate(uploaded_files):
            st.markdown(f"**📄 Page {idx+1} — {uploaded.name}**")
            edited_text = st.text_area(
                f"text_{idx}",
                all_texts[idx],
                height=250,
                key=f"text_{idx}",
                label_visibility="collapsed"
            )
            all_texts[idx] = edited_text
            st.markdown("---")

    # TAB 3: DOWNLOAD
    with tab3:
        st.markdown("### 📥 Download Your Files")
        st.markdown("")

        col_d1, col_d2 = st.columns(2)

        if output_type in ["Text PDF (searchable)", "Both"]:
            text_pdf_path = f"{WORK_DIR}/text_output.pdf"
            if len(all_texts) == 1:
                text_to_pdf(all_texts[0], text_pdf_path)
            else:
                multi_page_pdf(all_texts, text_pdf_path)

            with col_d1:
                with open(text_pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Text PDF",
                        f,
                        file_name="scanned_text.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        if output_type in ["Image PDF (scan)", "Both"]:
            img_pdf_path = f"{WORK_DIR}/image_output.pdf"
            images_to_pdf(all_cropped_paths, img_pdf_path)

            with col_d2:
                with open(img_pdf_path, "rb") as f:
                    st.download_button(
                        "🖼️ Download Image PDF",
                        f,
                        file_name="scanned_images.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# ---------- EMPTY STATE ----------
else:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.2rem; color: #4c51bf; font-weight: 600; margin-bottom: 0.5rem;">
            Upload your document to get started
        </div>
        <div style="color: #64748b; font-size: 0.95rem;">
            Supports JPG, PNG, BMP, TIFF • Multiple files allowed
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Powerful Features")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">✂️</div>
            <div class="feature-title">Auto Page Detection</div>
            <div class="feature-desc">Smart cropping — sirf document part, extra background hata diya jaata hai</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Auto-Rotate</div>
            <div class="feature-desc">Tilted pages automatically detect karke seedhe kiye jaate hain</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌐</div>
            <div class="feature-title">Hindi + English OCR</div>
            <div class="feature-desc">Dono bhashayein ek saath — Hinglish documents bhi perfect</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Image Enhancement</div>
            <div class="feature-desc">Denoise, sharpen, contrast boost — best OCR quality</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📑</div>
            <div class="feature-title">Multi-Page PDF</div>
            <div class="feature-desc">Ek saath kai images upload karo, ek PDF me merge ho jayengi</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Table Detection</div>
            <div class="feature-desc">Documents me tables automatic detect karke highlight</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.9rem;">
    <div style="font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">📄 DocScan Pro</div>
    <div>Powered by Streamlit • Tesseract OCR • OpenCV</div>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7;">
        Made with ❤️ for Hindi + English documents
    </div>
</div>
""", unsafe_allow_html=True)
