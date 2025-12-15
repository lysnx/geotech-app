"""
PDF Report Generation for Geotechnical Analysis.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def generate_pdf_report(
    results: Dict,
    soil_params: Dict[str, float],
    foundation_params: Dict[str, float],
    analysis_params: Dict[str, Any],
    output_path: Path,
    plot_paths: Dict[str, Path] = None
) -> Path:
    """
    Generate a professional PDF report of the analysis.
    
    Note: Requires reportlab library. Falls back to text report if unavailable.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Geotechnical Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Soil Properties
        story.append(Paragraph("Soil Properties", styles['Heading2']))
        soil_data = [["Parameter", "Value", "Unit"]]
        param_units = {
            "cohesion_c": ("Effective Cohesion", "kPa"),
            "friction_angle_phi": ("Friction Angle", "degrees"),
            "unit_weight_sat": ("Saturated Unit Weight", "kN/m3"),
            "poissons_ratio": ("Poisson's Ratio", "-"),
            "consolidation_coeff": ("Consolidation Coefficient", "m2/year"),
            "ocr": ("OCR", "-"),
        }
        for key, (name, unit) in param_units.items():
            if key in soil_params:
                soil_data.append([name, f"{soil_params[key]:.3f}", unit])
        
        soil_table = Table(soil_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        soil_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(soil_table)
        story.append(Spacer(1, 20))
        
        # Foundation
        story.append(Paragraph("Foundation Configuration", styles['Heading2']))
        found_data = [["Parameter", "Value", "Unit"]]
        found_data.append(["Width (B)", f"{foundation_params.get('width_B', 0):.2f}", "m"])
        found_data.append(["Length (L)", f"{foundation_params.get('length_L', 0):.2f}", "m"])
        found_data.append(["Depth (D)", f"{foundation_params.get('depth_D', 0):.2f}", "m"])
        found_data.append(["Applied Stress", f"{foundation_params.get('applied_stress_q', 0):.1f}", "kPa"])
        
        found_table = Table(found_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        found_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(found_table)
        story.append(Spacer(1, 20))
        
        # Critical Results
        min_fs = float('inf')
        critical_time = 0
        critical_depth = 0
        for t in results:
            for z in results[t]:
                fs = results[t][z].get('fs', float('inf'))
                if fs < min_fs:
                    min_fs = fs
                    critical_time = t
                    critical_depth = z
        
        story.append(Paragraph("Critical Analysis Results", styles['Heading2']))
        status = "SAFE" if min_fs > 1.5 else ("MARGINAL" if min_fs > 1.0 else "CRITICAL")
        story.append(Paragraph(f"Minimum Safety Factor: {min_fs:.3f} ({status})", styles['Normal']))
        story.append(Paragraph(f"Critical Depth: {critical_depth:.2f} m", styles['Normal']))
        story.append(Paragraph(f"Critical Time: {critical_time} days", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add plots if available
        if plot_paths:
            story.append(Paragraph("Analysis Plots", styles['Heading2']))
            for name, path in plot_paths.items():
                if path.exists():
                    story.append(Paragraph(name.replace('_', ' ').title(), styles['Heading3']))
                    story.append(Image(str(path), width=5*inch, height=4*inch))
                    story.append(Spacer(1, 10))
        
        doc.build(story)
        return output_path
        
    except ImportError:
        # Fallback to text report
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write("GEOTECHNICAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("SOIL PROPERTIES\n")
            for k, v in soil_params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nFOUNDATION\n")
            for k, v in foundation_params.items():
                f.write(f"  {k}: {v}\n")
        return text_path
