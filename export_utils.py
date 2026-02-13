"""
Export utilities for CardioGAM-Fusion++ Dashboard
Supports PDF, CSV, and Excel export formats
"""
import pandas as pd
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from flask import send_file
import io

def export_to_csv(data, filename_prefix="cardio_fusion_export"):
    """
    Export data to CSV format

    Args:
        data: List of dictionaries or pandas DataFrame
        filename_prefix: Prefix for the filename

    Returns:
        Tuple of (filename, file_content)
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    # Convert to CSV
    csv_content = df.to_csv(index=False)

    return filename, csv_content

def export_to_excel(data, filename_prefix="cardio_fusion_export"):
    """
    Export data to Excel format

    Args:
        data: List of dictionaries or pandas DataFrame
        filename_prefix: Prefix for the filename

    Returns:
        Tuple of (filename, file_content)
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.xlsx"

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Patient Data', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Patient Data']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    excel_content = output.getvalue()

    return filename, excel_content

def create_pdf_report(patient_data, risk_assessment, recommendations, filename_prefix="cardio_fusion_report"):
    """
    Create a comprehensive PDF medical report

    Args:
        patient_data: Dictionary with patient information
        risk_assessment: Dictionary with risk assessment results
        recommendations: String with clinical recommendations
        filename_prefix: Prefix for the filename

    Returns:
        Tuple of (filename, pdf_content)
    """
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.pdf"

    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        textColor=colors.darkblue
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 12
    normal_style.spaceAfter = 12

    # Build PDF content
    content = []

    # Title
    content.append(Paragraph("CardioGAM-Fusion++ Medical Report", title_style))
    content.append(Spacer(1, 12))

    # Report generation info
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    content.append(Paragraph("AI-Powered Cardiovascular Risk Assessment System", normal_style))
    content.append(Spacer(1, 24))

    # Patient Information Section
    content.append(Paragraph("Patient Information", subtitle_style))

    patient_table_data = [
        ["Patient ID", patient_data.get('patient_id', 'N/A')],
        ["Age", f"{patient_data.get('age', 'N/A')} years"],
        ["Blood Pressure", f"{patient_data.get('bp', 'N/A')} mmHg"],
        ["Cholesterol", f"{patient_data.get('cholesterol', 'N/A')} mg/dL"],
        ["Heart Rate", f"{patient_data.get('heart_rate', 'N/A')} bpm"],
    ]

    patient_table = Table(patient_table_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(patient_table)
    content.append(Spacer(1, 24))

    # Risk Assessment Results Section
    content.append(Paragraph("Risk Assessment Results", subtitle_style))

    risk_table_data = [
        ["Risk Score", f"{risk_assessment.get('risk_score', 'N/A'):.3f}"],
        ["Risk Category", risk_assessment.get('risk_category', 'N/A')],
        ["Confidence Level", f"{risk_assessment.get('confidence', 'N/A'):.1f}%"],
        ["Assessment Date", risk_assessment.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))],
    ]

    risk_table = Table(risk_table_data, colWidths=[2*inch, 3*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    content.append(risk_table)
    content.append(Spacer(1, 24))

    # Clinical Recommendations Section
    content.append(Paragraph("Clinical Recommendations", subtitle_style))
    content.append(Paragraph(recommendations, normal_style))
    content.append(Spacer(1, 24))

    # Footer
    content.append(Paragraph("This report was generated by CardioGAM-Fusion++ AI system.", normal_style))
    content.append(Paragraph("Please consult with a healthcare professional for medical decisions.", normal_style))

    # Build PDF
    doc.build(content)
    buffer.seek(0)
    pdf_content = buffer.getvalue()

    return filename, pdf_content

def create_comprehensive_report(patient_id, include_ecg_data=False):
    """
    Create a comprehensive medical report for a patient

    Args:
        patient_id: Patient ID to generate report for
        include_ecg_data: Whether to include ECG analysis data

    Returns:
        Tuple of (filename, pdf_content)
    """
    from src.dashboard.models import PatientAssessment

    # Get patient data
    assessment = PatientAssessment.query.filter_by(patient_id=patient_id).order_by(PatientAssessment.created_at.desc()).first()

    if not assessment:
        raise ValueError(f"No assessment found for patient {patient_id}")

    patient_data = {
        'patient_id': assessment.patient_id,
        'age': assessment.age,
        'bp': assessment.bp,
        'cholesterol': assessment.cholesterol,
        'heart_rate': assessment.heart_rate
    }

    risk_assessment = {
        'risk_score': assessment.risk_score,
        'risk_category': assessment.risk_category,
        'confidence': assessment.confidence,
        'created_at': assessment.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }

    recommendations = assessment.recommendations

    return create_pdf_report(patient_data, risk_assessment, recommendations, f"comprehensive_report_{patient_id}")

def export_patient_history(patient_id, format_type="csv"):
    """
    Export complete patient history

    Args:
        patient_id: Patient ID to export history for
        format_type: Export format ('csv', 'excel', 'pdf')

    Returns:
        Tuple of (filename, file_content)
    """
    from src.dashboard.models import PatientAssessment

    # Get all assessments for patient
    assessments = PatientAssessment.query.filter_by(patient_id=patient_id).order_by(PatientAssessment.created_at).all()

    if not assessments:
        raise ValueError(f"No history found for patient {patient_id}")

    # Convert to data format
    data = []
    for assessment in assessments:
        data.append({
            'Patient ID': assessment.patient_id,
            'Assessment Date': assessment.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Age': assessment.age,
            'Blood Pressure': assessment.bp,
            'Cholesterol': assessment.cholesterol,
            'Heart Rate': assessment.heart_rate,
            'Risk Score': assessment.risk_score,
            'Risk Category': assessment.risk_category,
            'Confidence': assessment.confidence,
            'Recommendations': assessment.recommendations
        })

    if format_type == "csv":
        return export_to_csv(data, f"patient_history_{patient_id}")
    elif format_type == "excel":
        return export_to_excel(data, f"patient_history_{patient_id}")
    elif format_type == "pdf":
        # Create summary PDF for patient history
        if data:
            latest = data[-1]  # Most recent assessment
            patient_data = {
                'patient_id': latest['Patient ID'],
                'age': latest['Age'],
                'bp': latest['Blood Pressure'],
                'cholesterol': latest['Cholesterol'],
                'heart_rate': latest['Heart Rate']
            }
            risk_assessment = {
                'risk_score': latest['Risk Score'],
                'risk_category': latest['Risk Category'],
                'confidence': latest['Confidence'],
                'created_at': latest['Assessment Date']
            }
            recommendations = latest['Recommendations']
            return create_pdf_report(patient_data, risk_assessment, recommendations, f"patient_summary_{patient_id}")
    else:
        raise ValueError(f"Unsupported format: {format_type}")
