import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

pdf_path = r"C:\Users\LENOVO\Desktop\HalluciGuard_System_Architecture_and_Stakeholders.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=40, leftMargin=40,
    topMargin=40, bottomMargin=40
)

styles = getSampleStyleSheet()

# Color Palette
primary_color = colors.HexColor("#0F172A")  # Slate 900
accent_color = colors.HexColor("#4F46E5")   # Indigo 600
secondary_color = colors.HexColor("#0284C7")# Light Blue 600
text_dark = colors.HexColor("#1E293B")      # Slate 800
text_muted = colors.HexColor("#475569")     # Slate 600
bg_light = colors.HexColor("#F8FAFC")       # Slate 50
bg_card = colors.HexColor("#F1F5F9")        # Slate 100
border_color = colors.HexColor("#CBD5E1")   # Slate 300

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=20,
    leading=24,
    textColor=primary_color,
    alignment=TA_LEFT,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    fontName='Helvetica',
    fontSize=10.5,
    leading=14,
    textColor=accent_color,
    spaceAfter=12
)

h1_style = ParagraphStyle(
    'H1',
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=17,
    textColor=primary_color,
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'H2',
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=14,
    textColor=accent_color,
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body',
    fontName='Helvetica',
    fontSize=9,
    leading=13.5,
    textColor=text_dark,
    spaceAfter=5,
    alignment=TA_LEFT
)

body_bold = ParagraphStyle(
    'BodyBold',
    parent=body_style,
    fontName='Helvetica-Bold'
)

code_box_style = ParagraphStyle(
    'CodeBox',
    fontName='Courier',
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#0F172A"),
    spaceAfter=4
)

table_header = ParagraphStyle(
    'TableHeader',
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11.5,
    textColor=colors.white,
    alignment=TA_LEFT
)

table_cell = ParagraphStyle(
    'TableCell',
    fontName='Helvetica',
    fontSize=8,
    leading=11,
    textColor=text_dark
)

table_cell_bold = ParagraphStyle(
    'TableCellBold',
    fontName='Helvetica-Bold',
    fontSize=8,
    leading=11,
    textColor=primary_color
)

story = []

# --- Title Header ---
story.append(Paragraph("HalluciGuard Multi-Agent Platform Specification", title_style))
story.append(Paragraph("Target Users, Enterprise Stakeholders & Proposed Overall Architecture", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=10))

# --- Section 1: Target Users & Stakeholders ---
story.append(Paragraph("1. Target Users & Enterprise Stakeholders Analysis", h1_style))
intro_users = (
    "The HalluciGuard platform caters to two primary groups: <b>Target Users</b> (who directly operate, integrate, "
    "or consume the system) and <b>Stakeholders</b> (who manage business risk, compliance budgets, and enterprise SLA policies)."
)
story.append(Paragraph(intro_users, body_style))

# Target Users Table
data_users = [
    [Paragraph("Target User Group", table_header), Paragraph("Operational Role", table_header), Paragraph("Primary Value Received", table_header)],
    [Paragraph("Enterprise AI & ML Engineers", table_cell_bold), Paragraph("Integrate HalluciGuard into RAG & LLM pipelines via SDK / REST API.", table_cell), Paragraph("Out-of-the-box safety guardrail with sub-millisecond overhead (0.2ms).", table_cell)],
    [Paragraph("Human Reviewers / SME Experts", table_cell_bold), Paragraph("Handle safety-critical ESCALATE_HUMAN flags (Doctors, Analysts).", table_cell), Paragraph("Filters 95%+ of routine queries; presents structured audit evidence for edge cases.", table_cell)],
    [Paragraph("End-Users & Consumers", table_cell_bold), Paragraph("Interact with enterprise chatbots, health QA & internal knowledge bots.", table_cell), Paragraph("Zero exposure to harmful, incorrect, or hallucinated AI information.", table_cell)],
]
t_users = Table(data_users, colWidths=[120, 190, 220])
t_users.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, primary_color),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t_users)
story.append(Spacer(1, 8))

# Stakeholders Table
data_sh = [
    [Paragraph("Stakeholder Category", table_header), Paragraph("Business / Organizational Focus", table_header), Paragraph("Why HalluciGuard is Essential", table_header)],
    [Paragraph("CTO & Chief AI Officer (CAIO)", table_cell_bold), Paragraph("System reliability, vendor cost control, architectural SLAs.", table_cell), Paragraph("Eliminates expensive LLM-as-a-Judge API calls; prevents silent pipeline failures.", table_cell)],
    [Paragraph("Chief Risk Officer (CRO) & Legal", table_cell_bold), Paragraph("Preventing lawsuits, regulatory fines & brand damage.", table_cell), Paragraph("Enforces domain policies (Healthcare/Finance); provides 100% reproducible audit logs.", table_cell)],
    [Paragraph("B2B Enterprise Clients", table_cell_bold), Paragraph("Requiring strict SLAs on factual grounding & safety.", table_cell), Paragraph("Tangible proof of enterprise-grade AI governance before signing contracts.", table_cell)],
    [Paragraph("Regulatory Bodies (EU AI Act, FDA)", table_cell_bold), Paragraph("Enforcing AI transparency, auditability & human oversight.", table_cell), Paragraph("Aligns directly with NIST AI RMF, HIPAA, SEC, and EU AI Act requirements.", table_cell)],
]
t_sh = Table(data_sh, colWidths=[120, 190, 220])
t_sh.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), accent_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, accent_color),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t_sh)
story.append(Spacer(1, 10))

# --- Section 2: Proposed Solution & Overall Architecture ---
story.append(Paragraph("2. Proposed Solution — Overall System Architecture", h1_style))
arch_intro = (
    "<b>HalluciGuard</b> is a decoupled, multi-agent cooperative platform. Instead of relying on a single prompt check, "
    "five autonomous agents work in a synchronized pipeline to detect, verify, remember, repair, and govern LLM responses."
)
story.append(Paragraph(arch_intro, body_style))

# End-to-End Diagram Box
diag_lines = [
    "[User Query + Draft Response] --> [1. DETECTOR AGENT] (Scans & scores hallucination probability)",
    "                                          │",
    "                                          v",
    "                                [2. VERIFIER AGENT] (Fetches grounding facts from FDA/SEC/NVD)",
    "                                          │",
    "                                          v",
    "                                [3. MEMORY AGENT]   (Queries historical records & reliability)",
    "                                          │",
    "                                          v",
    "                                [4. JUDGE AGENT (CDO)] (12-phase pipeline & domain governance)",
    "                                          │",
    "         ┌────────────────────────────────┼────────────────────────────────┐",
    "         v                                v                                v",
    "   [ACCEPT]                        [CORRECT]                       [ESCALATE_HUMAN]",
    "Release to User             [5. CORRECTOR AGENT]                Escalate to Human Expert",
    "                            Repairs invalid claims"
]

diag_text = "<br/>".join([f"<font face='Courier' size='7'>{line.replace(' ', '&nbsp;')}</font>" for line in diag_lines])

data_diag = [[Paragraph(diag_text, body_style)]]
t_diag = Table(data_diag, colWidths=[530])
t_diag.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,0), bg_card),
    ('BOX', (0,0), (-1,-1), 1, border_color),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(t_diag)
story.append(Spacer(1, 10))

# --- The 5 Core Agents Breakdown ---
story.append(Paragraph("The 5 Core Autonomous Agents", h2_style))

agents_data = [
    [Paragraph("Agent Name", table_header), Paragraph("Role Title", table_header), Paragraph("Key Technical Responsibilities", table_header)],
    [Paragraph("1. Detector Agent", table_cell_bold), Paragraph("Risk Auditor", table_cell), Paragraph("Performs initial fast scan; computes Hallucination Probability (0.0–1.0) & extracts candidate claims.", table_cell)],
    [Paragraph("2. Verifier Agent", table_cell_bold), Paragraph("Fact Researcher", table_cell), Paragraph("Fetches external grounding facts from FDA, SEC 10-K, NVD, PubMed; builds claim-evidence pairs.", table_cell)],
    [Paragraph("3. Memory Agent", table_cell_bold), Paragraph("Chief Historian", table_cell), Paragraph("Tracks recurring hallucination patterns, maintains source reliability index, stores past outcomes.", table_cell)],
    [Paragraph("4. Corrector Agent", table_cell_bold), Paragraph("Repair Engineer", table_cell), Paragraph("Rewrites hallucinated claims using verified evidence (e.g. fixing $500B -> $383.3B) while preserving tone.", table_cell)],
    [Paragraph("5. Judge Agent", table_cell_bold), Paragraph("Chief Decision Officer (CDO)", table_cell), Paragraph("Executes 12-phase pipeline (NLI, conflict classification, runtime health); enforces domain policy; issues final decision (ACCEPT, CORRECT, VERIFY_AGAIN, REJECT, ESCALATE_HUMAN, ABSTAIN).", table_cell)],
]
t_agents = Table(agents_data, colWidths=[100, 90, 340])
t_agents.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, primary_color),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t_agents)
story.append(Spacer(1, 10))

# --- Section 3: Architectural Differentiators ---
story.append(Paragraph("3. Key Architectural Differentiators", h1_style))

diff_data = [
    [Paragraph("Architectural Dimension", table_header), Paragraph("Traditional Guardrails", table_header), Paragraph("HalluciGuard Framework", table_header)],
    [Paragraph("System Architecture", table_cell_bold), Paragraph("Monolithic single-prompt LLM checker.", table_cell), Paragraph("Decoupled 5-Agent Cooperative Ecosystem.", table_cell)],
    [Paragraph("Decision Logic", table_cell_bold), Paragraph("Basic numerical threshold (if score > 0.7).", table_cell), Paragraph("Context-Aware Domain Governance Policies.", table_cell)],
    [Paragraph("Latency & Overhead", table_cell_bold), Paragraph("High (1–3 seconds via external LLM APIs).", table_cell), Paragraph("Sub-Millisecond (0.2ms avg) via Local NLI.", table_cell)],
    [Paragraph("Conflict Classification", table_cell_bold), Paragraph("Swallows errors or returns binary pass/fail.", table_cell), Paragraph("Categorizes Direct, Numeric, Temporal & Safety conflicts.", table_cell)],
    [Paragraph("Granularity & Auditability", table_cell_bold), Paragraph("Entire response level pass/fail.", table_cell), Paragraph("Per-Claim Risk Score (0%-100%) + Immutable JSON Audit Trail.", table_cell)],
]
t_diff = Table(diff_data, colWidths=[120, 195, 215])
t_diff.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), secondary_color),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
    ('BOX', (0,0), (-1,-1), 1, secondary_color),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t_diff)

doc.build(story)
print(f"SUCCESS: System Architecture PDF generated at {pdf_path}")
