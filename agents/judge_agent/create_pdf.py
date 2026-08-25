import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

pdf_path = r"C:\Users\LENOVO\Desktop\HalluciGuard_Presentation_Script_and_Guide.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=40, leftMargin=40,
    topMargin=40, bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor("#1E1B4B")  # Deep Indigo
accent_color = colors.HexColor("#4F46E5")   # Vibrant Indigo
text_dark = colors.HexColor("#1F2937")      # Charcoal
text_muted = colors.HexColor("#4B5563")     # Slate Gray
bg_light = colors.HexColor("#F3F4F6")       # Light Cool Gray
bg_card = colors.HexColor("#EEF2FF")        # Soft Blue/Indigo Tint
border_color = colors.HexColor("#C7D2FE")

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=primary_color,
    alignment=TA_LEFT,
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    fontName='Helvetica',
    fontSize=11,
    leading=15,
    textColor=accent_color,
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'H1',
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    textColor=primary_color,
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'H2',
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=15,
    textColor=accent_color,
    spaceBefore=10,
    spaceAfter=6,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body',
    fontName='Helvetica',
    fontSize=9.5,
    leading=14,
    textColor=text_dark,
    spaceAfter=6,
    alignment=TA_LEFT
)

body_bold = ParagraphStyle(
    'BodyBold',
    parent=body_style,
    fontName='Helvetica-Bold'
)

script_speak = ParagraphStyle(
    'ScriptSpeak',
    fontName='Helvetica-Oblique',
    fontSize=9.5,
    leading=14.5,
    textColor=colors.HexColor("#1E293B"),
    spaceAfter=6
)

table_header = ParagraphStyle(
    'TableHeader',
    fontName='Helvetica-Bold',
    fontSize=9,
    leading=12,
    textColor=colors.white,
    alignment=TA_LEFT
)

table_cell = ParagraphStyle(
    'TableCell',
    fontName='Helvetica',
    fontSize=8.5,
    leading=11.5,
    textColor=text_dark
)

table_cell_bold = ParagraphStyle(
    'TableCellBold',
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11.5,
    textColor=primary_color
)

story = []

# --- Header Title ---
story.append(Paragraph("HalluciGuard Judge Agent", title_style))
story.append(Paragraph("Enterprise Presentation Script, Target Audience Strategy & Technical Defense Guide", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=14))

# --- Section 1: Executive Concept ---
story.append(Paragraph("1. What is the 'Chief Decision Officer' (CDO)?", h1_style))
cdo_text = (
    "In traditional AI guardrails, a 'Judge' is merely a numerical threshold calculator (e.g. <i>if risk > 0.7, block response</i>). "
    "In HalluciGuard, the <b>Judge Agent is an AI Operating System acting as the Chief Decision Officer (CDO)</b>.<br/><br/>"
    "Like a C-suite executive who does not write code or balance ledgers directly, the CDO does <b>not</b> recalculate raw hallucination scores "
    "(the Detector's job) nor fetch external facts (the Verifier's job). Instead, it inspects outputs from all agents, evaluates pipeline health, "
    "checks source authority, enforces domain laws, resolves conflicts, and issues binding, audit-ready verdicts."
)
story.append(Paragraph(cdo_text, body_style))

# C-Suite Analogy Table
data_csuite = [
    [Paragraph("Agent Role", table_header), Paragraph("Corporate Equivalent", table_header), Paragraph("Core Responsibility", table_header)],
    [Paragraph("Detector Agent", table_cell_bold), Paragraph("Chief Auditor", table_cell), Paragraph("Measures raw risk & hallucination probabilities.", table_cell)],
    [Paragraph("Verifier Agent", table_cell_bold), Paragraph("Head of Research", table_cell), Paragraph("Retrieves external facts, documentation & citations.", table_cell)],
    [Paragraph("Memory Agent", table_cell_bold), Paragraph("Chief Historian", table_cell), Paragraph("Tracks past errors, recurring hallucination patterns & source records.", table_cell)],
    [Paragraph("Corrector Agent", table_cell_bold), Paragraph("Repair Engineer", table_cell), Paragraph("Rewrites and repairs invalid or ungrounded claims.", table_cell)],
    [Paragraph("Judge Agent", table_cell_bold), Paragraph("Chief Decision Officer", table_cell), Paragraph("Orchestrates 12-phase pipeline & issues binding verdict.", table_cell)],
]
t_csuite = Table(data_csuite, colWidths=[110, 110, 310])
t_csuite.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, primary_color),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t_csuite)
story.append(Spacer(1, 14))

# --- Section 2: Target Audience Strategy ---
story.append(Paragraph("2. Target Audience Analysis & Strategy", h1_style))

data_aud = [
    [Paragraph("Audience Segment", table_header), Paragraph("Their Primary Pain Point", table_header), Paragraph("Winning Value Proposition", table_header)],
    [Paragraph("Enterprise AI Architects", table_cell_bold), Paragraph("Latency overhead, non-deterministic outputs, API costs.", table_cell), Paragraph("Sub-millisecond latency (0.2ms avg), deterministic domain policies, zero API token costs.", table_cell)],
    [Paragraph("Governance & Risk Officers", table_cell_bold), Paragraph("Safety violations, regulatory fines, medical/financial harm.", table_cell), Paragraph("Domain strictness (VERY_STRICT vs RELAXED), safety conflict escalation, 100% audit trail.", table_cell)],
    [Paragraph("Product Managers & Devs", table_cell_bold), Paragraph("Complex integration, poor developer experience, slow builds.", table_cell), Paragraph("Clean Flask REST API (/evaluate), 1-line Python integration, live real-time web dashboard.", table_cell)],
    [Paragraph("Hackathon / VC Evaluators", table_cell_bold), Paragraph("Generic wrapper apps, lack of technical depth & novelty.", table_cell), Paragraph("Novel 'AI Operating System' architecture, 12-phase pipeline, 100% benchmark calibration.", table_cell)],
]
t_aud = Table(data_aud, colWidths=[120, 190, 220])
t_aud.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, primary_color),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t_aud)
story.append(Spacer(1, 14))

# --- Section 3: Q&A Defense Strategy ---
story.append(Paragraph("3. Anticipated Q&A Questions & Technical Answers", h1_style))

q1 = "<b>Q1: Why not just use LLM-as-a-Judge (e.g. a GPT-4 prompt)?</b><br/>" \
     "<i>Answer:</i> LLM-as-a-Judge suffers from the exact same failure modes it is trying to evaluate—hallucination, high cost, and high latency (1–3 seconds). " \
     "HalluciGuard's CDO runs in <b>sub-milliseconds (0.2ms)</b>, enforces strict deterministic domain policies, classifies specific conflict types (numeric/safety), " \
     "and produces an immutable audit record without calling external LLM APIs for governance."
story.append(Paragraph(q1, body_style))

q2 = "<b>Q2: How does the Judge handle conflicting evidence from multiple sources?</b><br/>" \
     "<i>Answer:</i> The Judge uses a dedicated <b>Conflict Resolver</b> and <b>Source Reliability Analyzer</b>. " \
     "If a community source (like Wikipedia) conflicts with an official source (like an FDA drug label or SEC 10-K filing), the CDO prioritizes the authoritative source. " \
     "If two official sources contradict on a safety claim, the CDO immediately triggers <b>ESCALATE_HUMAN</b> to prevent safety risk."
story.append(Paragraph(q2, body_style))
story.append(Spacer(1, 14))

# --- Section 4: 2-Minute Presentation Script ---
story.append(Paragraph("4. 2-Minute Verbal Presentation Script", h1_style))
story.append(Paragraph("<b>Target Speaking Rate:</b> ~130–140 words/min | <b>Total Length:</b> ~270 words (~2 minutes)", subtitle_style))

script_box = [
    [Paragraph("<b>[0:00 - 0:25] The Hook & The Problem</b>", table_cell_bold)],
    [Paragraph("\"Good morning everyone. As Large Language Models enter production in mission-critical industries—like Healthcare, Finance, and Cybersecurity—the biggest bottleneck isn't generating answers... it's trust.<br/><br/>"
               "Traditional hallucination guardrails use simple numerical thresholds. But numbers alone don't prevent catastrophic failures, like recommending a dangerous drug interaction or releasing an incorrect financial figure. That’s why we built the <b>HalluciGuard Judge Agent</b>.\"", script_speak)],
    [Paragraph("<b>[0:25 - 0:55] Core Philosophy: The Chief Decision Officer</b>", table_cell_bold)],
    [Paragraph("\"We completely redesigned the Judge Agent. It is <b>not</b> another confidence score calculator. It does not duplicate what the Detector or Verifier already do.<br/><br/>"
               "Instead, the Judge Agent acts as the <b>Chief Decision Officer (CDO)</b> of HalluciGuard—an AI Operating System that reasons about evidence quality, system health, source authority, and policy compliance before releasing a response.\"", script_speak)],
    [Paragraph("<b>[0:55 - 1:30] Technical Architecture: 12-Phase Pipeline</b>", table_cell_bold)],
    [Paragraph("\"Under the hood, the Judge Agent executes a <b>12-Phase Governance Pipeline</b>. First, it loads domain-specific policies—for instance, Healthcare policy strictly disallows unverified community sources and instantly escalates safety contradictions.<br/><br/>"
               "Second, it runs claim-level NLI entailment and numeric mismatch detection to compute per-claim hallucination risk scores. Finally, it maps out one of six actionable verdicts—whether to <b>ACCEPT</b>, <b>CORRECT</b>, <b>VERIFY_AGAIN</b>, <b>REJECT</b>, or <b>ESCALATE</b> to a human reviewer—producing an immutable, audit-ready decision record.\"", script_speak)],
    [Paragraph("<b>[1:30 - 2:00] Real-World Impact & Demo</b>", table_cell_bold)],
    [Paragraph("\"In our enterprise benchmark across 11 complex real-world scenarios in Healthcare, Finance, and Security, the Judge Agent achieved a <b>100% decision calibration pass rate</b> with sub-millisecond latency.<br/><br/>"
               "We also built an interactive web dashboard and benchmark suite so teams can visually simulate decisions in real time. Thank you, and I’m ready for your questions!\"", script_speak)]
]

t_script = Table(script_box, colWidths=[530])
t_script.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,0), bg_card),
    ('BACKGROUND', (0,2), (0,2), bg_card),
    ('BACKGROUND', (0,4), (0,4), bg_card),
    ('BACKGROUND', (0,6), (0,6), bg_card),
    ('BACKGROUND', (0,1), (0,1), bg_light),
    ('BACKGROUND', (0,3), (0,3), bg_light),
    ('BACKGROUND', (0,5), (0,5), bg_light),
    ('BACKGROUND', (0,7), (0,7), bg_light),
    ('BOX', (0,0), (-1,-1), 1, border_color),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('RIGHTPADDING', (0,0), (-1,-1), 10),
]))
story.append(t_script)

doc.build(story)
print(f"SUCCESS: PDF generated at {pdf_path}")
