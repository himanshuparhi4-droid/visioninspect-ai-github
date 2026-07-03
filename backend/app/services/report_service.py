from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.models.inspection_model import Inspection
from app.services.cloudinary_service import upload_image_or_local_url
from app.services.prediction_service import uploads_path


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm


def draw_wrapped_text(pdf: canvas.Canvas, text: str, x: float, y: float, max_width: float, font: str, size: int) -> float:
    pdf.setFont(font, size)
    words = str(text).split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if pdf.stringWidth(candidate, font, size) <= max_width:
            line = candidate
        else:
            pdf.drawString(x, y, line)
            y -= size + 4
            line = word
    if line:
        pdf.drawString(x, y, line)
        y -= size + 4
    return y


def draw_label_value(pdf: canvas.Canvas, label: str, value: str, x: float, y: float, width: float) -> None:
    pdf.setFillColor(colors.HexColor("#667085"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(x, y + 14, label.upper())
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(x, y, str(value))
    pdf.setStrokeColor(colors.HexColor("#dbe2ea"))
    pdf.line(x, y - 7, x + width, y - 7)


def draw_status_badge(pdf: canvas.Canvas, label: str, x: float, y: float, color: str) -> None:
    pdf.setFillColor(colors.HexColor(color))
    pdf.roundRect(x, y, 42 * mm, 10 * mm, 3 * mm, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(x + 21 * mm, y + 3.2 * mm, label)


def draw_image_if_exists(pdf: canvas.Canvas, image_path: str | None, x: float, y: float, width: float, height: float) -> None:
    if image_path and Path(image_path).exists():
        pdf.drawImage(image_path, x, y, width=width, height=height, preserveAspectRatio=True, anchor="c")
    else:
        pdf.setFillColor(colors.HexColor("#f4f6f8"))
        pdf.rect(x, y, width, height, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#667085"))
        pdf.setFont("Helvetica", 10)
        pdf.drawCentredString(x + width / 2, y + height / 2, "Image unavailable")


def inspection_report_path(inspection: Inspection) -> Path:
    return uploads_path("reports").joinpath(f"inspection_{inspection.id}.pdf")


def generate_inspection_report_pdf(inspection: Inspection) -> Path:
    report_path = inspection_report_path(inspection)
    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    pdf.setTitle(f"VisionInspect AI Report - {inspection.id}")

    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.rect(0, PAGE_HEIGHT - 34 * mm, PAGE_WIDTH, 34 * mm, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 16 * mm, "VisionInspect AI")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 24 * mm, "Manufacturing Defect Detection and Quality Inspection Report")

    badge_color = "#168a5b"
    if inspection.pass_fail == "Fail":
        badge_color = "#c93535"
    elif inspection.pass_fail == "Review":
        badge_color = "#b76b05"
    draw_status_badge(pdf, inspection.pass_fail or "Pending", PAGE_WIDTH - MARGIN - 42 * mm, PAGE_HEIGHT - 22 * mm, badge_color)

    y = PAGE_HEIGHT - 48 * mm
    col_width = (PAGE_WIDTH - 2 * MARGIN - 10 * mm) / 2
    draw_label_value(pdf, "Inspection ID", str(inspection.id), MARGIN, y, col_width)
    draw_label_value(pdf, "Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), MARGIN + col_width + 10 * mm, y, col_width)

    y -= 22 * mm
    draw_label_value(pdf, "Prediction", inspection.prediction or "Pending", MARGIN, y, col_width)
    draw_label_value(pdf, "Defect Type", inspection.defect_type or "Unknown", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 22 * mm
    confidence = f"{inspection.confidence * 100:.1f}%" if inspection.confidence is not None else "Pending"
    draw_label_value(pdf, "Confidence", confidence, MARGIN, y, col_width)
    draw_label_value(pdf, "Model Version", inspection.model_version or "Pending", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 22 * mm
    draw_label_value(pdf, "Severity Score", inspection.severity_score or "Pending", MARGIN, y, col_width)
    draw_label_value(pdf, "Severity Level", inspection.severity_level or "Pending", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 18 * mm
    draw_label_value(pdf, "Product ID", inspection.product_id or "Unassigned", MARGIN, y, col_width)
    draw_label_value(pdf, "Batch Number", inspection.batch_number or "Unassigned", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 18 * mm
    draw_label_value(pdf, "Line / Shift", f"{inspection.production_line or 'Unassigned'} / {inspection.shift or 'Unassigned'}", MARGIN, y, col_width)
    draw_label_value(pdf, "Source", inspection.source_label or inspection.source_type, MARGIN + col_width + 10 * mm, y, col_width)

    y -= 18 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "Visual Evidence")

    image_y = y - 68 * mm
    image_w = col_width
    image_h = 58 * mm
    draw_image_if_exists(pdf, inspection.original_image_path, MARGIN, image_y, image_w, image_h)
    draw_image_if_exists(pdf, inspection.heatmap_path, MARGIN + col_width + 10 * mm, image_y, image_w, image_h)
    pdf.setFillColor(colors.HexColor("#667085"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(MARGIN, image_y - 6 * mm, "Original image")
    pdf.drawString(MARGIN + col_width + 10 * mm, image_y - 6 * mm, "Defect heatmap")

    pdf.setFillColor(colors.HexColor("#667085"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(MARGIN, 12 * mm, "Generated by VisionInspect AI - automated AI result requires production policy review before shipment decisions.")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 12 * mm, "Page 1 of 2")

    pdf.showPage()
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.rect(0, PAGE_HEIGHT - 24 * mm, PAGE_WIDTH, 24 * mm, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 15 * mm, "Severity and Quality Decision")

    y = PAGE_HEIGHT - 42 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "Severity Components")
    y -= 12 * mm
    components = inspection.severity_components or {}
    for label, key in [
        ("Defect size", "size_score"),
        ("Location risk", "location_score"),
        ("Defect type risk", "defect_type_score"),
        ("Confidence score", "confidence_score"),
    ]:
        draw_label_value(pdf, label, components.get(key, "Pending"), MARGIN, y, col_width)
        y -= 15 * mm

    y -= 2 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "Recommended Action")
    y -= 9 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    y = draw_wrapped_text(
        pdf,
        inspection.recommended_action or "Manual quality review recommended.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        "Helvetica",
        11,
    )

    y -= 8 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "Quality Summary")
    y -= 12 * mm
    draw_label_value(pdf, "Pass / Fail Decision", inspection.pass_fail or "Pending", MARGIN, y, col_width)
    draw_label_value(pdf, "Review Status", inspection.review_status or "Pending", MARGIN + col_width + 10 * mm, y, col_width)
    y -= 22 * mm
    draw_label_value(pdf, "Anomaly Score", inspection.anomaly_score or "Pending", MARGIN, y, col_width)
    draw_label_value(pdf, "Defect Area Ratio", inspection.defect_area_ratio or "Pending", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 22 * mm
    explainability = inspection.explainability or {}
    draw_label_value(pdf, "Heatmap P95", explainability.get("heatmap_intensity_p95", "Pending"), MARGIN, y, col_width)
    draw_label_value(pdf, "Critical Location", "Yes" if explainability.get("critical_location") else "No", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 22 * mm
    draw_label_value(pdf, "Operator", inspection.operator_name or "Unassigned", MARGIN, y, col_width)
    draw_label_value(pdf, "Reviewed At", inspection.reviewed_at or "Pending", MARGIN + col_width + 10 * mm, y, col_width)

    y -= 18 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "AI Explainability Notes")
    y -= 9 * mm
    notes = explainability.get("notes") or ["No explainability notes recorded."]
    for note in notes:
        y = draw_wrapped_text(pdf, f"- {note}", MARGIN, y, PAGE_WIDTH - 2 * MARGIN, "Helvetica", 10)

    y -= 6 * mm
    pdf.setFillColor(colors.HexColor("#18212f"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "Review Notes")
    y -= 9 * mm
    y = draw_wrapped_text(
        pdf,
        inspection.review_notes or "No reviewer notes recorded.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        "Helvetica",
        11,
    )

    pdf.setFillColor(colors.HexColor("#667085"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(MARGIN, 12 * mm, "Generated by VisionInspect AI - automated AI result requires production policy review before shipment decisions.")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 12 * mm, "Page 2 of 2")
    pdf.save()

    return report_path


def generate_inspection_report_file(inspection: Inspection) -> str:
    report_path = generate_inspection_report_pdf(inspection)
    return upload_image_or_local_url(report_path, "reports")
