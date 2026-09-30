"""
HalluciGuard - Reviewer Q&A Dossier PDF generator.

Self-contained: reportlab (layout) + matplotlib mathtext (formula rendering).
No MS Word / pandoc / LaTeX dependency. Content lives in reviewer_content.py.

Run:  python scripts/build_reviewer_pdf.py
Out:  docs/reviewer/HalluciGuard_Reviewer_QA.pdf
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "reviewer"
OUT_PDF = OUT_DIR / "HalluciGuard_Reviewer_QA.pdf"
FORMULA_DIR = PROJECT_ROOT / "data" / "_formulas"
FORMULA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
NAVY = colors.HexColor("#1A365D")
TEAL = colors.HexColor("#0D9488")
BLUE = colors.HexColor("#2563EB")
DARK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#475569")
BG_LIGHT = colors.HexColor("#F8FAFC")
BG_CODE = colors.HexColor("#F1F5F9")
BORDER = colors.HexColor("#CBD5E1")
GREEN = colors.HexColor("#16A34A")
RED = colors.HexColor("#DC2626")
AMBER = colors.HexColor("#D97706")
PURPLE = colors.HexColor("#7C3AED")
WHITE = colors.white

# ---------------------------------------------------------------- styles
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.8, leading=14,
                      textColor=DARK, spaceAfter=5)
BODY_S = ParagraphStyle("bodyS", parent=BODY, fontSize=9.0, leading=12.5)
Q_STYLE = ParagraphStyle("q", fontName="Helvetica-Bold", fontSize=12.5, leading=15,
                         textColor=NAVY, spaceBefore=4, spaceAfter=4)
SEC_STYLE = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=15, leading=18,
                           textColor=WHITE)
BULLET = ParagraphStyle("bullet", parent=BODY, fontSize=9.3, leading=12.8, spaceAfter=2)
CALL_TITLE = ParagraphStyle("ct", fontName="Helvetica-Bold", fontSize=9.8, leading=13,
                            spaceAfter=2)
CALL_BODY = ParagraphStyle("cb", parent=BODY, fontSize=9.2, leading=12.8, spaceAfter=0)
COVER_T = ParagraphStyle("covT", fontName="Helvetica-Bold", fontSize=30, leading=34,
                         textColor=NAVY, spaceAfter=6)
COVER_SUB = ParagraphStyle("covS", fontName="Helvetica", fontSize=12.5, leading=17,
                           textColor=MUTED, spaceAfter=4)
COVER_KICK = ParagraphStyle("covK", fontName="Helvetica-Bold", fontSize=11, leading=14,
                            textColor=TEAL, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11, textColor=DARK)
CELL_B = ParagraphStyle("cellB", parent=CELL, fontName="Helvetica-Bold", textColor=WHITE)


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------- math -> png
def _sanitize_latex(src: str) -> str:
    """Make a doc formula renderable by matplotlib mathtext."""
    s = src.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    # \boxed{X} -> {X}  (grouping braces render invisibly)
    s = s.replace(r"\boxed", "")
    # mathtext has no \text; \mathrm renders upright text
    s = s.replace(r"\text", r"\mathrm")
    return s


def render_formula(latex: str, fontsize: int = 15) -> Image | None:
    """Render a LaTeX-ish string to a transparent PNG; return a reportlab Image.
    Returns None if mathtext cannot parse it (caller falls back to text box)."""
    s = _sanitize_latex(latex)
    key = hashlib.md5((s + f"|{fontsize}").encode()).hexdigest()[:16]
    png = FORMULA_DIR / f"f_{key}.png"
    if not png.exists():
        try:
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0, 0, f"${s}$", fontsize=fontsize, color="#0F172A")
            fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.06,
                        transparent=True)
            plt.close(fig)
        except Exception:
            plt.close("all")
            return None
    from PIL import Image as PILImage
    with PILImage.open(png) as im:
        w, h = im.size
    scale = 200 / 72.0  # dpi -> pt
    w_pt, h_pt = w / scale, h / scale
    max_w = 150 * mm
    if w_pt > max_w:
        h_pt *= max_w / w_pt
        w_pt = max_w
    return Image(str(png), width=w_pt, height=h_pt)


# ---------------------------------------------------------------- flowables
def section_banner(title: str, subtitle: str = "", bg=NAVY):
    inner = [Paragraph(esc(title), SEC_STYLE)]
    if subtitle:
        inner.append(Paragraph(esc(subtitle),
                     ParagraphStyle("sb", fontName="Helvetica", fontSize=9,
                                    textColor=colors.HexColor("#D9E2EC"), leading=12)))
    t = Table([[inner]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return KeepTogether([Spacer(1, 6), t, Spacer(1, 8)])


def q_header(num: int, text: str):
    bar = Table([[Paragraph(f"<b>Q{num}.</b>&nbsp;&nbsp;{esc(text)}", Q_STYLE)]],
                colWidths=[170 * mm])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return bar


def body(text: str):
    return Paragraph(text, BODY)


def bullets(items):
    lis = []
    for it in items:
        lis.append(ListItem(Paragraph(it, BULLET), leftIndent=6, value="•"))
    return ListFlowable(lis, bulletType="bullet", start="•", leftIndent=10,
                        bulletFontSize=8, bulletColor=TEAL)


def formula_box(latex: str, caption: str = ""):
    img = render_formula(latex)
    if img is None:
        content = Paragraph(f"<font face='Courier'>{esc(_sanitize_latex(latex))}</font>",
                            ParagraphStyle("fx", parent=BODY, alignment=TA_CENTER,
                                           fontName="Courier", fontSize=10))
    else:
        content = img
    cells = [[content]]
    if caption:
        cells.append([Paragraph(esc(caption),
                      ParagraphStyle("fc", parent=BODY_S, alignment=TA_CENTER,
                                     textColor=MUTED, fontSize=8))])
    t = Table(cells, colWidths=[170 * mm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FBFDFF")),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]
    t.setStyle(TableStyle(style))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 4)])


def callout(title: str, text: str, accent=NAVY):
    tp = ParagraphStyle("ctc", parent=CALL_TITLE, textColor=accent)
    inner = [Paragraph("■ " + esc(title), tp), Paragraph(text, CALL_BODY)]
    t = Table([[inner]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 4)])


_STATUS = {
    "CONFIRMED": GREEN, "VERIFIED": GREEN, "PASS": GREEN, "SUPPORTED": GREEN,
    "MATCH": GREEN, "TRUE": GREEN, "YES": GREEN,
    "CONTRADICTED": RED, "WRONG": RED, "FAIL": RED, "FALSE": RED, "NO": RED,
    "UNVERIFIED": AMBER, "NOT_ENOUGH_INFO": AMBER, "PARTIAL": AMBER,
    "HISTORICAL": AMBER, "NEI": AMBER, "ABSTAIN": AMBER,
    "CONFLICTED": PURPLE, "NOT ESTABLISHED": RED,
}


def styled_table(headers, rows, col_widths):
    data = [[Paragraph(esc(str(h)), CELL_B) for h in headers]]
    for row in rows:
        data.append([Paragraph(esc(str(c)), CELL) for c in row])
    t = Table(data, colWidths=[w * mm for w in col_widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), BG_LIGHT))
        # color status keyword cells
        for c in range(len(headers)):
            val = str(rows[r - 1][c]).strip().upper()
            if val in _STATUS:
                data[r][c] = Paragraph(f"<b>{esc(str(rows[r-1][c]))}</b>",
                                       ParagraphStyle(f"s{r}{c}", parent=CELL,
                                                      textColor=_STATUS[val]))
    t.setStyle(TableStyle(style))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 5)])


def code_block(txt: str):
    p = Paragraph(esc(txt).replace("\n", "<br/>").replace(" ", "&nbsp;"),
                  ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=11,
                                 textColor=colors.HexColor("#1E293B")))
    t = Table([[p]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_CODE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 4)])


# ---------------------------------------------------------------- page frame
def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    # top rule
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.4)
    canvas.line(20 * mm, h - 14 * mm, w - 20 * mm, h - 14 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(20 * mm, h - 12.5 * mm, "HALLUCIGUARD")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 20 * mm, h - 12.5 * mm, "Reviewer Preparation Dossier")
    # footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 14 * mm, w - 20 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm,
                      "Repository-grounded. Verified against source code before publication.")
    canvas.drawRightString(w - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _blank(canvas, doc):
    pass


def cover_story(meta_rows):
    st = []
    st.append(Spacer(1, 40))
    st.append(Paragraph("REVIEWER PREPARATION DOSSIER", COVER_KICK))
    st.append(Paragraph("HalluciGuard", COVER_T))
    st.append(Paragraph(
        "A verification &amp; governance layer around LLM answers &mdash; questions, "
        "repository-grounded answers, and the exact scoring formulas, prepared for "
        "technical review.", COVER_SUB))
    st.append(Spacer(1, 10))
    st.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    st.append(Spacer(1, 14))
    st.append(styled_table(["Specification", "Value (as configured in the repository)"],
                           meta_rows, [55, 115]))
    st.append(Spacer(1, 12))
    st.append(callout(
        "How to read this dossier",
        "Every technical claim below was checked against the actual HalluciGuard source "
        "code and configuration. Where a common assumption is wrong (for example that "
        "the deployment uses Nginx, or that a legacy 100% benchmark reflects current "
        "full-system accuracy), the answer states the correction explicitly. Formulas are "
        "reproduced exactly as implemented in the code.", TEAL))
    st.append(PageBreak())
    return st


def build(content, meta_rows):
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
        title="HalluciGuard - Reviewer Q&A Dossier",
        author="HalluciGuard")
    story = cover_story(meta_rows)
    for block in content:
        kind = block[0]
        if kind == "section":
            story.append(section_banner(block[1], block[2] if len(block) > 2 else "",
                         block[3] if len(block) > 3 else NAVY))
        elif kind == "q":
            story.append(KeepTogether([q_header(block[1], block[2]), Spacer(1, 3)]))
        elif kind == "body":
            story.append(body(block[1]))
        elif kind == "bullets":
            story.append(bullets(block[1]))
            story.append(Spacer(1, 3))
        elif kind == "formula":
            story.append(formula_box(block[1], block[2] if len(block) > 2 else ""))
        elif kind == "callout":
            story.append(callout(block[1], block[2],
                         block[3] if len(block) > 3 else NAVY))
        elif kind == "table":
            story.append(styled_table(block[1], block[2], block[3]))
        elif kind == "code":
            story.append(code_block(block[1]))
        elif kind == "spacer":
            story.append(Spacer(1, block[1]))
        elif kind == "pagebreak":
            story.append(PageBreak())
    doc.build(story, onFirstPage=_blank, onLaterPages=_decorate)
    print(f"[+] Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    from reviewer_content import CONTENT, META_ROWS
    build(CONTENT, META_ROWS)



