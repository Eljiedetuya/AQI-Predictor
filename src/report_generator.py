from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(prediction, mae, r2):
    doc = SimpleDocTemplate("AQI_Report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI AQI Prediction Report", styles['Title']))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(f"Predicted AQI: {prediction:.2f}", styles['Normal']))
    elements.append(Paragraph(f"Model MAE: {mae:.2f}", styles['Normal']))
    elements.append(Paragraph(f"Model R2: {r2:.2f}", styles['Normal']))

    doc.build(elements)