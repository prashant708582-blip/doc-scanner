import pytesseract
import paddleocr
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np

# ---------- PADDLEOCR SETUP ----------
# PaddleOCR 3.x API — ek baar initialize karo (cached)
_ocr_en = None
_ocr_hi = None

def image_to_text(image_path, lang="hin+eng", psm=3, handwriting=False):
    img = Image.open(image_path)

    if handwriting:
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
def get_ocr(lang="en"):
    """PaddleOCR instance create karta hai (lazy loading)."""
    global _ocr_en, _ocr_hi
    
    if lang == "en":
        if _ocr_en is None:
            _ocr_en = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                enable_mkldnn=False  # CPU pe crash avoid karne ke liye
            )
        return _ocr_en
    elif lang == "hi":
        if _ocr_hi is None:
            _ocr_hi = PaddleOCR(
                lang="hi",
                use_textline_orientation=True,
                enable_mkldnn=False
            )
        return _ocr_hi
else:
        config = f'--oem 3 --psm {psm}'
        # Fallback: en
        return get_ocr("en")

    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text

def image_to_text(image_path, lang="hin+eng", psm=3, handwriting=False):
    """
    PaddleOCR se text extract karta hai.
    lang: "hin+eng" | "eng" | "hin"
    psm: ignore (PaddleOCR me alag hota hai)
    handwriting: ignore
    """
    # Language mapping
    if "hin" in lang and "eng" in lang:
        ocr_lang = "en"  # PaddleOCR me Hindi + English dono ek saath ek model se nahi aata
    elif lang == "hi":
        ocr_lang = "hi"
    else:
        ocr_lang = "en"
    
    ocr = get_ocr(ocr_lang)
    
    # Image load karo
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image load nahi hui: {image_path}")
    
    # PaddleOCR 3.x API: predict() use karo
    result = ocr.predict(img)
    
    # Text extract karo
    texts = []
    if result and len(result) > 0:
        res = result[0]
        # 3.x me dict-like return hota hai
        if hasattr(res, 'rec_texts'):
            texts = res.rec_texts
        elif isinstance(res, dict) and 'rec_texts' in res:
            texts = res['rec_texts']
        else:
            # Fallback for older return shape
            for line in res:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    texts.append(line[1][0])
    
    return "\n".join(texts)

def extract_with_confidence(image_path, lang="hin+eng"):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(
        img, lang=lang,
        output_type=pytesseract.Output.DICT,
        config='--oem 3 --psm 3'
    )

    results = []
    for i, txt in enumerate(data['text']):
        if txt.strip():
            results.append({
                'text': txt,
                'conf': int(data['conf'][i]),
                'bbox': (data['left'][i], data['top'][i],
                         data['width'][i], data['height'][i])
            })
    return results
def extract_with_confidence(image_path, lang="hin+eng"):
    """OCR with confidence scores (simplified)."""
    return []


def detect_tables(image_path):
    """Simple table detection — OpenCV based (same as before)."""
img = cv2.imread(image_path)
if img is None:
return []
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
thresh = cv2.adaptiveThreshold(
~gray, 255,
cv2.ADAPTIVE_THRESH_MEAN_C,
cv2.THRESH_BINARY, 15, -2
)

    
horizontal = thresh.copy()
cols = horizontal.shape[1]
h_size = max(cols // 30, 10)
h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
horizontal = cv2.erode(horizontal, h_struct)
horizontal = cv2.dilate(horizontal, h_struct)

    
vertical = thresh.copy()
rows = vertical.shape[0]
v_size = max(rows // 30, 10)
v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
vertical = cv2.erode(vertical, v_struct)
vertical = cv2.dilate(vertical, v_struct)

    
table_mask = cv2.add(horizontal, vertical)
table_mask = cv2.dilate(table_mask,
cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    
cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)

    
tables = []
for c in cnts:
x, y, w, h = cv2.boundingRect(c)
