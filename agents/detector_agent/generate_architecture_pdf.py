"""PDF Generator script for HalluciGuard Detector Agent System Architecture Summary."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f172a")
    ACCENT_BLUE = colors.HexColor("#2563eb")
    ACCENT_DARK = colors.HexColor("#1e293b")
    TEXT_DARK = colors.HexColor("#334155")

    # Custom Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        fontName="Helvetica-Bold",
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=ACCENT_BLUE,
        fontName="Helvetica-Bold",
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=ACCENT_DARK,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        fontName="Helvetica",
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName="Helvetica-Bold"
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK,
        fontName="Helvetica"
    )

    # Title Banner
    story.append(Paragraph("HalluciGuard Detector Agent", title_style))
    story.append(Paragraph("System Architecture & Technical Specification | Version 2.0.0", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_BLUE, spaceAfter=10))

    # Executive Overview
    story.append(Paragraph("1. Executive System Overview", h2_style))
    overview_text = (
        "<b>HalluciGuard</b> is a multi-signal agentic AI system designed to detect and quantify hallucinations "
        "in Large Language Model (LLM) outputs. The <b>Detector Agent</b> synthesizes four statistical uncertainty signals: "
        "Token Probability, Predictive Entropy, Semantic Similarity, and conditionally gated Self-Consistency. "
        "An <b>Intelligent Gating Mechanism (SelfConsistencyGate)</b> triggers multi-sample stochastic decoding only "
        "when single-pass uncertainty evaluates to MEDIUM risk on analytical query categories, reducing compute overhead by over 75%."
    )
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 8))

    # Signal Matrix Table
    story.append(Paragraph("2. Detection Signal Specification", h2_style))
    signal_data = [
        [Paragraph("Signal Name", table_header_style), Paragraph("Underlying Metric / Formula", table_header_style), Paragraph("Weight", table_header_style), Paragraph("Target Hallucination Indicator", table_header_style)],
        [Paragraph("<b>Token Probability</b>", table_cell_style), Paragraph("S<sub>prob</sub> = exp(avg_logprob)", table_cell_style), Paragraph("0.35", table_cell_style), Paragraph("Low average token log-probability across response.", table_cell_style)],
        [Paragraph("<b>Predictive Entropy</b>", table_cell_style), Paragraph("S<sub>entropy</sub> = 1.0 - (H / ln V)", table_cell_style), Paragraph("0.25", table_cell_style), Paragraph("High Shannon entropy over token vocabulary distribution.", table_cell_style)],
        [Paragraph("<b>Semantic Similarity</b>", table_cell_style), Paragraph("S<sub>sem</sub> = (cos(&theta;) + 1.0) / 2.0", table_cell_style), Paragraph("0.25", table_cell_style), Paragraph("Low embedding cosine similarity between prompt & response.", table_cell_style)],
        [Paragraph("<b>Self-Consistency</b>", table_cell_style), Paragraph("S<sub>sc</sub> = Pairwise Cosine Similarity", table_cell_style), Paragraph("0.15", table_cell_style), Paragraph("High variance / low agreement across N sampled completions.", table_cell_style)]
    ]
    sig_table = Table(signal_data, colWidths=[110, 160, 50, 220])
    sig_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 10))

    # Architecture Hierarchy Table
    story.append(Paragraph("3. Directory Structure & Module Hierarchy", h2_style))
    struct_data = [
        [Paragraph("Module Path", table_header_style), Paragraph("Primary Responsibility", table_header_style), Paragraph("Key Classes / Functions", table_header_style)],
        [Paragraph("detector_agent/detector.py", table_cell_style), Paragraph("Pipeline orchestrator for 4-signal aggregation", table_cell_style), Paragraph("DetectorAgent", table_cell_style)],
        [Paragraph("detector_agent/config.py", table_cell_style), Paragraph("Pydantic configuration & signal weight validator", table_cell_style), Paragraph("DetectorConfig, SignalWeights", table_cell_style)],
        [Paragraph("detector_agent/model_manager.py", table_cell_style), Paragraph("Singleton PyTorch LLM & SentenceTransformer loader", table_cell_style), Paragraph("ModelManager", table_cell_style)],
        [Paragraph("detector_agent/gate.py", table_cell_style), Paragraph("Intelligent gating controller for Self-Consistency", table_cell_style), Paragraph("SelfConsistencyGate", table_cell_style)],
        [Paragraph("detector_agent/classifier.py", table_cell_style), Paragraph("Heuristic rule-based prompt category classifier", table_cell_style), Paragraph("PromptClassifier", table_cell_style)],
        [Paragraph("detector_agent/datasets/", table_cell_style), Paragraph("Hugging Face benchmark streaming infrastructure", table_cell_style), Paragraph("BenchmarkExample, HaluEvalLoader, TruthfulQALoader", table_cell_style)],
        [Paragraph("detector_agent/evaluation/", table_cell_style), Paragraph("Classification, pairwise, ablation & dashboard suite", table_cell_style), Paragraph("Evaluator, PairwiseEvaluator, AblationEvaluator, DashboardGenerator", table_cell_style)]
    ]
    struct_table = Table(struct_data, colWidths=[150, 220, 170])
    struct_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT_DARK),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(struct_table)
    story.append(Spacer(1, 10))

    # Calibrated Thresholds Table
    story.append(Paragraph("4. Calibrated Thresholds & Risk Categorization", h2_style))
    thresh_data = [
        [Paragraph("Hallucination Probability", table_header_style), Paragraph("Assigned Risk Level", table_header_style), Paragraph("Recommended Pipeline Action", table_header_style)],
        [Paragraph("Prob &le; 0.40", table_cell_style), Paragraph("<b>LOW Risk</b>", table_cell_style), Paragraph("NextAction.ACCEPT (Bypass secondary verification)", table_cell_style)],
        [Paragraph("0.40 &lt; Prob &lt; 0.55", table_cell_style), Paragraph("<b>MEDIUM Risk</b>", table_cell_style), Paragraph("NextAction.VERIFY (Triggers Gated Self-Consistency & Verifier)", table_cell_style)],
        [Paragraph("Prob &ge; 0.55", table_cell_style), Paragraph("<b>HIGH Risk</b>", table_cell_style), Paragraph("NextAction.VERIFY (Escalate to Correction Agent)", table_cell_style)]
    ]
    thresh_table = Table(thresh_data, colWidths=[150, 150, 240])
    thresh_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(thresh_table)

    doc.build(story)
    print(f"[PDFGenerator] Generated PDF at {os.path.abspath(output_path)}")


if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "architecture_summary.pdf")
    generate_pdf(out_pdf)
