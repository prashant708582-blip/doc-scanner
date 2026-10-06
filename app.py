import streamlit as st
import os
import cv2
import numpy as np

from scanner import (
    crop_document, enhance_for_ocr, auto_rotate,
    image_to_text, detect_tables,
    text_to_pdf, multi_page_pdf, images_to_pdf
)

st.set_page_config(
    page_title="DocScan Pro — Hindi + English OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

WORK_DIR = "work"
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- THEME STATE ----------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------- THEME COLORS ----------
THEMES = {
    "dark": {
        "bg": "radial-gradient(ellipse at top, #1a1a2e 0%, #0f0f1e 50%, #000000 100%)",
        "text": "#e2e8f0",
        "text_muted": "#94a3b8",
        "accent": "#a78bfa",
        "accent2": "#f093fb",
        "sidebar_bg": "linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%)",
        "card_bg": "linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7))",
        "card_border": "rgba(102, 126, 234, 0.25)",
        "input_bg": "rgba(30, 41, 59, 0.8)",
        "input_border": "rgba(167, 139, 250, 0.4)",
        "upload_bg": "linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1))",
        "upload_border": "rgba(167, 139, 250, 0.5)",
        "hr": "linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.5), transparent)",
        "scroll_track": "#0f0f1e",
        "shadow": "rgba(102, 126, 234, 0.4)",
    },
    "light": {
        "bg": "linear-gradient(180deg, #f8fafc 0%, #ffffff 100%)",
        "text": "#0f172a",
        "text_muted": "#64748b",
        "accent": "#667eea",
        "accent2": "#764ba2",
        "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)",
        "card_bg": "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
        "card_border": "rgba(102, 126, 234, 0.3)",
        "input_bg": "#ffffff",
        "input_border": "rgba(102, 126, 234, 0.4)",
        "upload_bg": "linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%)",
        "upload_border": "rgba(102, 126, 234, 0.5)",
        "hr": "linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.4), transparent)",
        "scroll_track": "#f1f5f9",
        "shadow": "rgba(102, 126, 234, 0.2)",
    }
}

T = THEMES[st.session_state.theme]

# ---------- DYNAMIC CSS ----------
st.markdown(f"""
<style>
    .stApp {{
        background: {T['bg']};
        background-attachment: fixed;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: {T['text']} !important;
    }}
    
    .hero {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px {T['shadow']};
        border: 1px solid rgba(255,255,255,0.15);
    }}
    .hero h1 {{
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .hero p {{
        font-size: 1.1rem;
        margin-top: 0.6rem;
        color: rgba(255,255,255,0.95) !important;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(0,0,0,0.25);
        padding: 6px 16px;
        border-radius: 25px;
        font-size: 0.85rem;
        margin: 4px;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3);
        font-weight: 500;
    }}
    
    section[data-testid="stSidebar"] {{
        background: {T['sidebar_bg']};
        border-right: 1px solid {T['card_border']};
    }}
    section[data-testid="stSidebar"] h2 {{
        color: {T['accent']} !important;
        font-size: 1.1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {T['card_border']};
    }}
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] input {{
        background: {T['input_bg']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['input_border']} !important;
        border-radius: 10px !important;
    }}
    
    [data-testid="stFileUploader"] {{
        background: {T['upload_bg']};
        border: 2px dashed {T['upload_border']};
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {T['accent']};
        box-shadow: 0 0 40px {T['shadow']};
    }}
    
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-weight: 700;
        box-shadow: 0 8px 24px {T['shadow']};
        transition: all 0.3s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 32px {T['shadow']};
        color: white !important;
    }}
    
    .stat-card {{
        background: {T['card_bg']};
        padding: 1.5rem 1rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid {T['card_border']};
        transition: all 0.3s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-5px);
        border-color: {T['accent']};
        box-shadow: 0 10px 30px {T['shadow']};
    }}
    .stat-number {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: {T['text_muted']} !important;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .feat-card {{
        background: {T['card_bg']};
        padding: 1.8rem 1.5rem;
        border-radius: 18px;
        border: 1px solid {T['card_border']};
        height: 100%;
        transition: all 0.4s ease;
    }}
    .feat-card:hover {{
        transform: translateY(-8px);
        border-color: {T['accent']};
        box-shadow: 0 15px 40px {T['shadow']};
    }}
    .feat-icon {{ font-size: 2.5rem; margin-bottom: 0.8rem; display: block; }}
    .feat-title {{ font-size: 1.1rem; font-weight: 700; color: {T['accent']} !important; margin-bottom: 0.5rem; }}
    .feat-desc {{ font-size: 0.9rem; color: {T['text_muted']} !important; line-height: 1.6; }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {T['card_bg']};
        padding: 8px;
        border-radius: 14px;
        border: 1px solid {T['card_border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        color: {T['text_muted']};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }}
    
    .stTextArea textarea {{
        background: {T['input_bg']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['input_border']} !important;
        border-radius: 12px !important;
    }}
    
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {T['scroll_track']}; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }}
    
    hr {{
        border: none;
        height: 1px;
        background: {T['hr']};
        margin: 2rem 0;
    }}
    img {{ border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>📄 DocScan Pro</h1>
    <p>AI-Powered Document Scanner · Hindi + English OCR</p>
    <div style="margin-top: 1rem;">
        <span class="hero-badge">✨ Auto-Crop</span>
        <span class="hero-badge">🔍 Smart OCR</span>
        <span class="hero-badge">📑 PDF Export</span>
        <span class="hero-badge">🌗 Theme Toggle</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    # Theme Toggle
    st.markdown("## 🎨 Appearance")
    theme_choice = st.radio(
        "Theme",
        ["🌙 Dark", "☀️ Light"],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    new_theme = "dark" if "Dark" in theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Configuration")
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
        "🎨 Enhancement",
        ["auto", "bw", "grayscale", "color"],
        index=0
    )

    st.markdown("---")
    st.markdown("## 🔧 Advanced")
    handwriting = st.checkbox("✍️ Handwriting mode", value=False)
    auto_rot = st.checkbox("🔄 Auto-rotate pages", value=True)

    st.markdown("---")
    st.markdown("## 📑 Output Format")
    output_type = st.radio(
        "Choose",
        ["Both", "Text PDF (searchable)", "Image PDF (scan)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    theme_emoji = "🌙" if st.session_state.theme == "dark" else "☀️"
    st.caption(f"{theme_emoji} Current theme: **{st.session_state.theme.title()}**")

# ---------- UPLOAD ----------
st.markdown("### 📤 Upload Your Document")
uploaded_files = st.file_uploader(
    "Choose files",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ---------- MAIN LOGIC ----------
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded — processing started...")

    all_texts = []
    all_cropped_paths = []
    all_enhanced_paths = []

    progress = st.progress(0)
    status = st.empty()

    for idx, uploaded in enumerate(uploaded_files):
        status.info(f"🔄 Processing: {uploaded.name} ({idx+1}/{len(uploaded_files)})")
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
    status.success("✨ All pages processed successfully!")

    # ---------- STATS ----------
    st.markdown("### 📊 Summary")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    total_chars = sum(len(t) for t in all_texts)
    total_words = sum(len(t.split()) for t in all_texts)

    with col_s1:
        st.markdown(f'<div class="stat-card"><p class="stat-number">{len(uploaded_files)}</p><p class="stat-label">Pages</p></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="stat-card"><p class="stat-number">{total_words}</p><p class="stat-label">Words</p></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div class="stat-card"><p class="stat-number">{total_chars}</p><p class="stat-label">Characters</p></div>', unsafe_allow_html=True)
    with col_s4:
        st.markdown(f'<div class="stat-card"><p class="stat-number">{len(all_texts)}</p><p class="stat-label">Completed</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # ---------- TABS ----------
    tab1, tab2, tab3 = st.tabs(["📄 Visual Results", "📝 Extracted Text", "⬇️ Downloads"])

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
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 2rem; 
                background: {T['upload_bg']};
                border-radius: 20px; border: 2px dashed {T['upload_border']};
                margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.4rem; color: {T['accent']}; font-weight: 700; margin-bottom: 0.5rem;">
            Upload your document to get started
        </div>
        <div style="color: {T['text_muted']}; font-size: 1rem;">
            Supports JPG, PNG, BMP, TIFF · Multiple files allowed
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Powerful Features")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">✂️</span>
            <div class="feat-title">Auto Page Detection</div>
            <div class="feat-desc">Smart cropping — extra background aur blank space auto remove</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">🌗</span>
            <div class="feat-title">Theme Toggle</div>
            <div class="feat-desc">Dark aur Light mode — apni pasand ka theme chuno</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">🌐</span>
            <div class="feat-title">Hindi + English OCR</div>
            <div class="feat-desc">Dono bhashayein ek saath — Hinglish documents bhi perfect</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">🎨</span>
            <div class="feat-title">Image Enhancement</div>
            <div class="feat-desc">Denoise, sharpen, contrast boost — best OCR quality</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">📑</span>
            <div class="feat-title">Multi-Page PDF</div>
            <div class="feat-desc">Ek saath kai images — ek PDF me merge ho jayengi</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div class="feat-card">
            <span class="feat-icon">📊</span>
            <div class="feat-title">Table Detection</div>
            <div class="feat-desc">Documents me tables automatic detect karke highlight</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; color: {T['text_muted']}; font-size: 0.9rem;">
    <div style="font-size: 1.3rem; font-weight: 800; 
                background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;">
        📄 DocScan Pro
    </div>
    <div style="color: {T['text_muted']};">
        Powered by Streamlit · Tesseract OCR · OpenCV
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7; color: {T['text_muted']};">
        Made with ❤️ for Hindi + English documents
    </div>
</div>
""", unsafe_allow_html=True)
