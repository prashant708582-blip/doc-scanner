from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

FONT_PATHS = [
    "fonts/NotoSansDevanagari.ttf",
    "./fonts/NotoSansDevanagari.ttf",
    "/app/fonts/NotoSansDevanagari.ttf",
]


def register_fonts():
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('NotoDev', fp))
                return 'NotoDev'
            except Exception:
                continue
    return 'Helvetica'


def _wrap_line(line, max_chars):
    chunks = []
    while len(line) > max_chars:
        chunks.append(line[:max_chars])
        line = line[max_chars:]
    chunks.append(line)
    return chunks


def text_to_pdf(text, pdf_path):
    font = register_fonts()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 16
    max_chars = 90

    c.setFont(font, 11)
    y = height - margin

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(font, 11)
        y = height - margin

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line:
            y -= line_height
            if y < margin:
                new_page()
            continue

        for chunk in _wrap_line(line, max_chars):
            c.drawString(margin, y, chunk)
            y -= line_height
            if y < margin:
                new_page()

    c.save()
    return pdf_path


def multi_page_pdf(texts, pdf_path):
    font = register_fonts()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 16
    max_chars = 90

    for page_idx, text in enumerate(texts):
        if page_idx > 0:
            c.showPage()

        c.setFont(font, 11)
        y = height - margin

        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                y -= line_height
                if y < margin:
                    c.showPage()
                    c.setFont(font, 11)
                    y = height - margin
                continue

            for chunk in _wrap_line(line, max_chars):
                c.drawString(margin, y, chunk)
                y -= line_height
                if y < margin:
                    c.showPage()
                    c.setFont(font, 11)
                    y = height - margin

    c.save()
    return pdf_path


def images_to_pdf(image_paths, pdf_path):
    from PIL import Image
    images = []
    for p in image_paths:
        img = Image.open(p)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    if images:
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
    return pdf_path
