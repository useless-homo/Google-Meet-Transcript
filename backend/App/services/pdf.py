import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def generate_pdf(text, audio_path):
    os.makedirs("pdfs", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    pdf_path = f"pdfs/{base_name}_transcript.pdf"

    doc = SimpleDocTemplate(pdf_path)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("Meeting Transcript", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)

    return pdf_path
