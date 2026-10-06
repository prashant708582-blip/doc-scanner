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


def _draw_watermark(c, width, height, text):
    """Page pe watermark draw karta hai (diagonal)."""
    if not text:
        return
    c.saveState()
    c.setFont('Helvetica-Bold', 60)
    c.setFillColor(HexColor("#e2e8f0"), alpha=0.3)
    c.translate(width / 2, height / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def render_text_to_pdf(text, pdf_path, title="Scanned Document",
                        watermark="", header_text="", footer_text=""):
    """
    OCR text ko A4 white page pe proper format me likhta hai.
    Supports: title, watermark, header, footer.
    """
    font = register_fonts()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    margin_left = 25 * mm
    margin_right = 25 * mm
    margin_top = 25 * mm
    margin_bottom = 25 * mm

    body_size = 11
    line_height = 16

    usable_width = width - margin_left - margin_right
    max_chars = int(usable_width / (body_size * 0.55))

    page_number = 1

    def draw_page_decorations(page_num):
        # Watermark
        if watermark:
            _draw_watermark(c, width, height, watermark)

        # Header
        if header_text:
            c.setFont(font, 9)
            c.setFillColor(HexColor("#94a3b8"))
            c.drawString(margin_left, height - 12 * mm, header_text)
            c.setStrokeColor(HexColor("#e2e8f0"))
            c.setLineWidth(0.5)
            c.line(margin_left, height - 15 * mm, width - margin_right, height - 15 * mm)

        # Footer
        c.setFont(font, 8)
        c.setFillColor(HexColor("#94a3b8"))
        if footer_text:
            c.drawString(margin_left, 12 * mm, footer_text)
        c.drawCentredString(width / 2, 12 * mm, f"— Page {page_num} —")
        c.setFillColor(HexColor("#000000"))

    # Start first page
    c.setFont(font, body_size)
    c.setFillColor(HexColor("#000000"))
    y = height - margin_top

    # Draw decorations for first page
    draw_page_decorations(page_number)

    # Title
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

        if not line:
            y -= line_height * 0.6
            if y < margin_bottom + 20:
                c.showPage()
                page_number += 1
                draw_page_decorations(page_number)
                c.setFont(font, body_size)
                c.setFillColor(HexColor("#000000"))
                y = height - margin_top
            continue

        for chunk in _wrap_line(line, max_chars):
            if y < margin_bottom + 20:
                c.showPage()
                page_number += 1
                draw_page_decorations(page_number)
                c.setFont(font, body_size)
                c.setFillColor(HexColor("#000000"))
                y = height - margin_top

            c.drawString(margin_left, y, chunk)
            y -= line_height

    c.save()
    return pdf_path


def text_to_pdf(text, pdf_path, title="Scanned Document",
                 watermark="", header_text="", footer_text=""):
    return render_text_to_pdf(
        text, pdf_path, title=title,
        watermark=watermark, header_text=header_text, footer_text=footer_text
    )


def multi_page_pdf(texts, pdf_path, title="Multi-Page Document",
                    watermark="", header_text="", footer_text=""):
    """Multiple pages ka text — each text = one section."""
    combined = ""
    for i, t in enumerate(texts):
        combined += f"\n--- Page {i+1} ---\n"
        combined += t + "\n\n"
    return render_text_to_pdf(
        combined, pdf_path, title=title,
        watermark=watermark, header_text=header_text, footer_text=footer_text
    )


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


def add_password_to_pdf(input_pdf, output_pdf, password):
    """PDF pe password lagao."""
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        writer.encrypt(password)
        with open(output_pdf, "wb") as f:
            writer.write(f)
        return output_pdf
    except Exception as e:
        print(f"Password error: {e}")
        return input_pdf
