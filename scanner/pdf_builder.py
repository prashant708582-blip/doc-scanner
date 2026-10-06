from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
import os

FONT_PATHS = [
    "fonts/NotoSansDevanagari.ttf",
    "./fonts/NotoSansDevanagari.ttf",
    "/app/fonts/NotoSansDevanagari.ttf",
]


def register_fonts():
    """Hindi font register karta hai. Fallback Helvetica."""
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('NotoDev', fp))
                return 'NotoDev'
            except Exception:
                continue
    return 'Helvetica'


def _wrap_line(line, max_chars):
    """Long line ko chunks me todta hai."""
    chunks = []
    while len(line) > max_chars:
        chunks.append(line[:max_chars])
        line = line[max_chars:]
    chunks.append(line)
    return chunks


def render_text_to_pdf(text, pdf_path, title="Scanned Document"):
    """
    OCR text ko A4 white page pe proper format me likhta hai.
    - Auto page break
    - Empty lines handled
    - Long lines wrapped
    - Page numbers
    """
    font = register_fonts()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Margins (in points: 1 inch = 72 pt)
    margin_left = 25 * mm
    margin_right = 25 * mm
    margin_top = 25 * mm
    margin_bottom = 25 * mm

    # Font sizes
    body_size = 11
    line_height = 16

    # Text area width in characters (approx)
    usable_width = width - margin_left - margin_right
    # Estimate: avg char width = body_size * 0.5 for English, but Hindi is wider
    max_chars = int(usable_width / (body_size * 0.55))

    # Footer
    page_number = 1

    def draw_footer(page_num):
        c.setFont(font, 8)
        c.setFillColor(HexColor("#94a3b8"))
        c.drawCentredString(width / 2, 12 * mm, f"— Page {page_num} —")
        c.setFillColor(HexColor("#000000"))

    # Start first page
    c.setFont(font, body_size)
    c.setFillColor(HexColor("#000000"))
    y = height - margin_top

    # Title (optional)
    if title:
        c.setFont(font, 14)
        c.setFillColor(HexColor("#1e293b"))
        c.drawString(margin_left, y, title)
        y -= 20
        c.setStrokeColor(HexColor("#e2e8f0"))
        c.setLineWidth(0.5)
        c.line(margin_left, y, width - margin_right, y)
        y -= 20
        c.setFont(font, body_size)
        c.setFillColor(HexColor("#000000"))

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        # Empty line — chhota gap
        if not line:
            y -= line_height * 0.6
            if y < margin_bottom + 20:
                draw_footer(page_number)
                c.showPage()
                page_number += 1
                c.setFont(font, body_size)
                c.setFillColor(HexColor("#000000"))
                y = height - margin_top
            continue

        # Long lines wrap
        for chunk in _wrap_line(line, max_chars):
            if y < margin_bottom + 20:
                draw_footer(page_number)
                c.showPage()
                page_number += 1
                c.setFont(font, body_size)
                c.setFillColor(HexColor("#000000"))
                y = height - margin_top

            c.drawString(margin_left, y, chunk)
            y -= line_height

    # Last page footer
    draw_footer(page_number)
    c.save()
    return pdf_path


# Backward compatibility
def text_to_pdf(text, pdf_path):
    return render_text_to_pdf(text, pdf_path, title="Scanned Document")


def multi_page_pdf(texts, pdf_path):
    """Multiple pages ka text — each text = one section."""
    combined = ""
    for i, t in enumerate(texts):
        combined += f"\n--- Page {i+1} ---\n"
        combined += t + "\n\n"
    return render_text_to_pdf(combined, pdf_path, title="Multi-Page Document")


def images_to_pdf(image_paths, pdf_path):
    """Images ko PDF me daalo (scan look)."""
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
