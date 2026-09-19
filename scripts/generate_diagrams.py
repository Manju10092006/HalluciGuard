"""
HalluciGuard Master Documentation & Technical Architecture Generator.

Generates:
  1. 15+ high-resolution, publication-quality technical diagrams in docs/diagrams/
  2. HalluciGuard_Technical_Architecture_and_Verification_Documentation.docx (FAANG-grade technical architecture book)
  3. HalluciGuard_Technical_Architecture_and_Verification_Documentation.pdf (via Word automation)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
DOCS_DIR = PROJECT_ROOT / "docs" / "architecture"
OUTPUT_DOCX = PROJECT_ROOT / "HalluciGuard_Technical_Architecture_and_Verification_Documentation.docx"
OUTPUT_PDF = PROJECT_ROOT / "HalluciGuard_Technical_Architecture_and_Verification_Documentation.pdf"

DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Color Palette (Light Theme, Modern Engineering Design)
# -----------------------------------------------------------------------------
C_NAVY = "#1A365D"       # Primary Deep Navy
C_BLUE = "#2563EB"       # Accent Blue
C_TEAL = "#0D9488"       # Secondary Accent Teal
C_CYAN = "#0284C7"       # Accent Cyan
C_BG = "#FFFFFF"         # Pure White
C_CARD_BG = "#F8FAFC"    # Pale Ice / Slate Card Fill
C_CARD_BORDER = "#CBD5E1"# Subtle Border
C_TEXT_DARK = "#0F172A"  # Charcoal / Black Text
C_TEXT_MUTED = "#475569" # Slate Muted Text
C_GREEN = "#16A34A"      # Success / Verified
C_RED = "#DC2626"        # Danger / Contradicted
C_AMBER = "#D97706"      # Warning / Unverified
C_PURPLE = "#7C3AED"     # Special / Conflicted

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"]


def save_fig(fig, filename: str):
    path = DIAGRAMS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  [+] Saved diagram: {path.name}")


# =============================================================================
# 1. System Architecture Diagram (Fig 01)
# =============================================================================
def generate_fig01():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 90)
    ax.axis("off")

    # Header
    ax.text(70, 86, "HALLUCIGUARD — SYSTEM-LEVEL ARCHITECTURE", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_NAVY)
    ax.text(70, 82.5, "End-to-End Pipeline: OpenRouter LLM → HaluEval Detector → n8n Retrieval → BGE Reranker → DeBERTa NLI → 4-State Verdict",
            ha="center", va="center", fontsize=9.5, color=C_TEXT_MUTED)

    # Boxes definition helper
    def draw_card(x, y, w, h, title, subtitle, color, border_color=C_CARD_BORDER, badge=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      facecolor=color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.8, title, ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT_DARK)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.8, subtitle, ha="center", va="center", fontsize=8, color=C_TEXT_MUTED, multialignment="center")
        if badge:
            brect = patches.FancyBboxPatch((x + w - 16, y + h - 3.5), 14, 2.5, boxstyle="round,pad=0.2",
                                           facecolor=C_NAVY, edgecolor="none")
            ax.add_patch(brect)
            ax.text(x + w - 9, y + h - 2.2, badge, ha="center", va="center", fontsize=6.5, fontweight="bold", color="#FFF")

    def draw_arrow(x1, y1, x2, y2, label="", color=C_NAVY, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.8, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 1.2, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color=color)

    # 1. User & Base LLM
    draw_card(5, 62, 24, 15, "1. Base LLM Service", "OpenRouter / Qwen3-4B\n(Candidate Generation)", C_CARD_BG, C_BLUE, "OPENROUTER")
    ax.text(17, 58, "User Prompt", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_BLUE)
    draw_arrow(17, 62, 17, 54)

    # 2. Detector Agent
    draw_card(5, 38, 24, 16, "2. Detector Agent", "HaluEval DistilBERT\nBinary Classification\n[prob, conf, risk_tier]", C_CARD_BG, C_TEAL, "LOCAL INFERENCE")
    
    # 3. Risk Router
    draw_card(36, 42, 22, 12, "3. Risk Gate Router", "LOW (≤0.30) → Accept\nMED/HIGH (≥0.50) → Verify", C_CARD_BG, C_AMBER, "ROUTING GATE")
    draw_arrow(29, 46, 36, 46, "draft response")

    # Accept branch
    draw_card(36, 18, 22, 12, "Accept Path", "Direct Pass-Through\nLatency < 50ms", "#ECFDF5", C_GREEN, "PASSED")
    draw_arrow(47, 42, 47, 30, "LOW Risk", C_GREEN)

    # Verify branch -> Claim Decomposition
    draw_card(65, 58, 26, 16, "4. Claim Decomposer", "Proposition Extraction\nPronoun Resolution\nSVO Triples & Entities", C_CARD_BG, C_CYAN, "VERIFIER AGENT")
    draw_arrow(58, 48, 65, 62, "MED/HIGH Risk", C_RED)

    # 5. n8n Retrieval Webhook
    draw_card(98, 58, 36, 18, "5. n8n Retrieval V2", "Domain Switch (5 Domains)\nPrimary: Wikipedia, PubMed, NVD, arXiv, SEC\nQuality Gate & Tavily Fallback", C_CARD_BG, C_PURPLE, "n8n CLOUD")
    draw_arrow(91, 66, 98, 66, "normalized claims")

    # 6. BGE Reranker
    draw_card(98, 32, 36, 16, "6. BGE Reranker Large", "BAAI/bge-reranker-large\nCross-Encoder Semantic Relevance\nIndependent bge_score ∈ [0, 1]", C_CARD_BG, C_BLUE, "CUDA / PyTorch")
    draw_arrow(116, 58, 116, 48, "raw passages")

    # 7. DeBERTa NLI
    draw_card(65, 32, 26, 16, "7. DeBERTa-v3 NLI", "cross-encoder/nli-deberta-v3\nPremise=Evidence, Hyp=Claim\n[Entailment, Contradiction, Neutral]", C_CARD_BG, C_TEAL, "CUDA / PyTorch")
    draw_arrow(98, 40, 91, 40, "ranked evidence")

    # 8. Evidence Scorer
    draw_card(65, 8, 26, 16, "8. Evidence Scorer", "Relation Check + Authority Weight\n4-Class: Supp/Contra/Neut/Irrel\nCalibrated Trust & Confidence", C_CARD_BG, C_NAVY, "DECISION ENGINE")
    draw_arrow(78, 32, 78, 24, "NLI triples")

    # 9. 4-State Verdict Output
    draw_card(98, 8, 36, 16, "9. Final 4-State Verdict", "VERIFIED | CONTRADICTED\nUNVERIFIED | CONFLICTED\nTrace Provenance & Explanations", "#EFF6FF", C_NAVY, "DECISION OUTPUT")
    draw_arrow(91, 16, 98, 16, "scored claim reports")

    save_fig(fig, "fig01_system_architecture.png")


# =============================================================================
# 2. Detector Architecture & Risk Routing (Fig 02)
# =============================================================================
def generate_fig02():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "DETECTOR AGENT ARCHITECTURE & RISK ROUTING GATING", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "HaluEval Fine-Tuned DistilBERT Classifier with Informational Diagnostics & Fail-Soft Routing",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    # Input Box
    rect = patches.FancyBboxPatch((5, 45), 28, 20, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(19, 61, "DetectionInput", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(19, 53, "• user_query: str\n• llm_response: str\n• max_length: 384 tokens\n• Format: [CLS] Q [SEP] R [SEP]",
            ha="center", va="center", fontsize=8, color=C_TEXT_MUTED)

    # Classifier Box
    rect = patches.FancyBboxPatch((40, 42), 34, 26, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_TEAL, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(57, 64, "HaluEval DistilBERT Engine", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(57, 53, "Model: Manjunath2000006/halluciguard-detector\nBase: distilbert-base-uncased\nForward Pass: Logits → Softmax\n\nExecution Diagnostics:\n• detector_model_loaded: bool\n• detector_inference_executed: bool\n• detector_degraded: bool\n• detector_model_source: str",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    ax.annotate("", xy=(40, 55), xytext=(33, 55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # Output Schema Box
    rect = patches.FancyBboxPatch((82, 42), 43, 26, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_CYAN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(103.5, 64, "DetectionResult Schema", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(103.5, 52, "• confidence_score: float ∈ [0.0, 1.0]\n• hallucination_probability: float ∈ [0.0, 1.0]\n• risk_level: RiskLevel (LOW / MED / HIGH)\n• next_action: NextAction (Accept / Verify)\n• model_source: 'halueval-distilbert'\n• diagnostics (provenance + degraded flags)",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    ax.annotate("", xy=(82, 55), xytext=(74, 55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # Gating Thresholds
    ax.text(65, 34, "THREE-TIER RISK ROUTING POLICY", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NAVY)

    # 3 Tiers
    # Low
    rect = patches.FancyBboxPatch((8, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#ECFDF5", edgecolor=C_GREEN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(25, 24, "LOW RISK (≤ 0.30)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_GREEN)
    ax.text(25, 16, "Action: ACCEPT\n• Verifier Skipped\n• High factuality confidence\n• Output directly returned", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    # Medium
    rect = patches.FancyBboxPatch((48, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#FFFBEB", edgecolor=C_AMBER, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(65, 24, "MEDIUM RISK (0.30 - 0.50)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_AMBER)
    ax.text(65, 16, "Action: VERIFY\n• Verifier Triggered\n• Potential factual uncertainty\n• Full evidence grounding", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    # High
    rect = patches.FancyBboxPatch((88, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#FEF2F2", edgecolor=C_RED, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(105, 24, "HIGH RISK (≥ 0.50)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_RED)
    ax.text(105, 16, "Action: VERIFY\n• Verifier Mandatory\n• Elevated hallucination probability\n• Multi-source cross-checking", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    save_fig(fig, "fig02_detector_routing.png")


# =============================================================================
# 3. n8n Workflow Topology & Domain Routing (Fig 03)
# =============================================================================
def generate_fig03():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(70, 86, "N8N RETRIEVAL SERVICE V2 WORKFLOW TOPOLOGY", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(70, 82.5, "Multi-Domain Live Evidence Ingestion with Quality Gating & Tavily Extract Fallback",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_node(x, y, w, h, name, desc, color=C_CARD_BG, border=C_PURPLE):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.2, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)
        if desc:
            ax.text(x + w/2, y + h/2 - 0.8, desc, ha="center", va="center", fontsize=7, color=C_TEXT_MUTED, multialignment="center")

    # Entry Webhook
    draw_node(4, 52, 22, 14, "Webhook: Receive Claim", "POST /halluciguard-verify-v2\nX-API-Key Authentication\nExtracts claim & mode", C_CARD_BG, C_BLUE)
    draw_node(30, 52, 22, 14, "LLM: Analyze Claim", "OpenRouter / Qwen3-14B\nExtracts Queries & Entities\nClassifies Routing Domain", C_CARD_BG, C_TEAL)
    draw_node(56, 52, 20, 14, "Code: Build Context", "Builds Runtime Payload\nNormalizes Query Tokens\nSets Timeouts & Flags", C_CARD_BG, C_CYAN)

    ax.annotate("", xy=(30, 59), xytext=(26, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))
    ax.annotate("", xy=(56, 59), xytext=(52, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # Domain Switch
    draw_node(80, 52, 18, 14, "Domain Switch", "Rule-Based Router\n5 Domain Ports\n+ Fallback Port", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(80, 59), xytext=(76, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # 5 Domain Specific Primary Sources (Stacked)
    domains = [
        ("General", "Wikipedia Search API (REST & Action)", 74),
        ("Healthcare", "PubMed XML + OpenFDA Drug Labels", 61),
        ("Cybersecurity", "NIST NVD CVE 2.0 API", 48),
        ("AI Research", "arXiv E-Query API (Atom XML)", 35),
        ("Finance", "SEC EDGAR EFTS 10-K/8-K Search", 22),
    ]
    for dname, ddesc, dy in domains:
        draw_node(104, dy, 32, 10, f"{dname} Primary", ddesc, "#F8FAFC", C_CYAN)
        ax.annotate("", xy=(104, dy + 5), xytext=(98, 59), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=C_PURPLE))

    # Assemble Primary Evidence
    draw_node(74, 18, 24, 13, "Assemble Primary", "Extracts text snippets\nComputes token score\nAttaches provenance", C_CARD_BG, C_NAVY)
    for _, _, dy in domains:
        ax.annotate("", xy=(86, 31), xytext=(104, dy + 5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_CARD_BORDER))

    # Quality Gate
    draw_node(46, 18, 24, 13, "Quality Evaluation", "usable_count ≥ 2 ?\nrelevant_count ≥ 2 ?\nDetermines Tavily Need", "#FEF3C7", C_AMBER)
    ax.annotate("", xy=(70, 24.5), xytext=(74, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # Tavily Web Fallback
    draw_node(18, 18, 24, 13, "Tavily Web Fallback", "Advanced Search (multi-query)\nBasic Markdown Extract\nExtract Latency & Scores", "#EFF6FF", C_BLUE)
    ax.annotate("", xy=(42, 24.5), xytext=(46, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_AMBER))

    # Merge & Response
    draw_node(4, 18, 12, 24, "Response", "Merge & Dedup\nTrace Object\nJSON Return", "#ECFDF5", C_GREEN)
    ax.annotate("", xy=(16, 24.5), xytext=(18, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    save_fig(fig, "fig03_n8n_retrieval_topology.png")


# =============================================================================
# 4. Evidence Lifecycle State Machine (Fig 04)
# =============================================================================
def generate_fig04():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 70)
    ax.axis("off")

    ax.text(65, 66, "EVIDENCE LIFECYCLE & MUTUALLY EXCLUSIVE CLASSIFICATION", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 62.5, "Strict Invariant: sum(supporting, contradicting, neutral, irrelevant) == total_passages",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_state(x, y, w, h, title, desc, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 0.8, desc, ha="center", va="center", fontsize=7.5, color=C_TEXT_DARK, multialignment="center")

    # State 1: Ingestion
    draw_state(5, 36, 25, 18, "1. Ingested Passage", "• Raw text snippet\n• Source provenance URL\n• adapter_score hint\n• relevance_score = 0.0", C_CARD_BG, C_BLUE)
    
    # State 2: Relevance Gate
    draw_state(38, 36, 25, 18, "2. Relevance Gating", "• BGE Semantic Score\n• Gate Threshold: 0.20\n• If score < 0.20 →\n  Marked IRRELEVANT", C_CARD_BG, C_CYAN)
    ax.annotate("", xy=(38, 45), xytext=(30, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # State 3: NLI Verification
    draw_state(71, 36, 25, 18, "3. NLI Classification", "• DeBERTa-v3-base forward\n• P(Entail), P(Contra), P(Neut)\n• SVO Relation Check Override\n• Non-assertive myth filter", C_CARD_BG, C_TEAL)
    ax.annotate("", xy=(71, 45), xytext=(63, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # 4 Output Classes
    draw_state(104, 52, 22, 12, "SUPPORTING", "Entailment ≥ 0.35\n& Entail > Contra", "#ECFDF5", C_GREEN)
    draw_state(104, 36, 22, 12, "CONTRADICTING", "Contra ≥ 0.35\n& Contra > Entail", "#FEF2F2", C_RED)
    draw_state(104, 20, 22, 12, "NEUTRAL", "Informational context\nNo decisive polarity", "#FFFBEB", C_AMBER)
    draw_state(104, 4, 22, 12, "IRRELEVANT", "BGE score < 0.20\nExcluded from verdict", "#F1F5F9", C_CARD_BORDER)

    ax.annotate("", xy=(104, 58), xytext=(96, 48), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_GREEN))
    ax.annotate("", xy=(104, 42), xytext=(96, 45), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_RED))
    ax.annotate("", xy=(104, 26), xytext=(96, 42), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_AMBER))
    ax.annotate("", xy=(104, 10), xytext=(63, 39), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_CARD_BORDER))

    save_fig(fig, "fig04_evidence_lifecycle.png")


# =============================================================================
# 5. Four-State Verdict Decision Tree (Fig 05)
# =============================================================================
def generate_fig05():
    fig, ax = plt.subplots(figsize=(13, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 85)
    ax.axis("off")

    ax.text(65, 81, "FOUR-STATE VERDICT DECISION TREE & THRESHOLD POLICIES", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 77.5, "Deterministic Python Verdict Engine (VERIFIED, CONTRADICTED, UNVERIFIED, CONFLICTED)",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_box(x, y, w, h, title, desc, color, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.2, title, ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 0.8, desc, ha="center", va="center", fontsize=7.5, color=C_TEXT_DARK, multialignment="center")

    # Root Evidence Evaluation
    draw_box(45, 60, 40, 14, "Evidence Scoring Inputs", "support_score = 0.70·max + 0.30·avg + bonus\ncontra_score = 0.70·max + 0.30·avg + bonus\nEffective Weight = Rel · Cred · Recency", C_CARD_BG, C_BLUE)

    # Decision Node 1: Conflicted
    draw_box(45, 40, 40, 14, "Conflict Evaluation", "support_score ≥ 0.30 AND\ncontra_score ≥ 0.30 AND\n|support - contra| < 0.15 ?", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(65, 54), xytext=(65, 60), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # Conflicted Output
    draw_box(95, 40, 30, 14, "CONFLICTED", "Credible conflicting evidence\nfrom authoritative sources\nE.g., active scientific debate", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(95, 47), xytext=(85, 47), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_PURPLE))
    ax.text(90, 49, "YES", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_PURPLE)

    # Decision Node 2: Polarity Check
    draw_box(45, 20, 40, 14, "Polarity Dominance Check", "support_score ≥ 0.30 & supp > contra ?\nOR\ncontra_score ≥ 0.25 & contra > supp ?", C_CARD_BG, C_CYAN)
    ax.annotate("", xy=(65, 34), xytext=(65, 40), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))
    ax.text(68, 37, "NO", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_NAVY)

    # 3 Terminal Verdicts from Polarity Check
    # Verified
    draw_box(95, 22, 30, 12, "VERIFIED", "support_score ≥ 0.30\nHigh evidence grounding\nZero contradiction", "#ECFDF5", C_GREEN)
    ax.annotate("", xy=(95, 28), xytext=(85, 28), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_GREEN))
    ax.text(90, 30, "Support", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_GREEN)

    # Contradicted
    draw_box(95, 6, 30, 12, "CONTRADICTED", "contra_score ≥ 0.25\nExplicit refutation found\nOr relation mismatch", "#FEF2F2", C_RED)
    ax.annotate("", xy=(95, 12), xytext=(85, 24), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_RED))
    ax.text(90, 18, "Contra", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_RED)

    # Unverified
    draw_box(5, 20, 30, 14, "UNVERIFIED", "Safe Abstention\nInsufficient evidence / low relevance\nRefuses to hallucinate truth", "#FFFBEB", C_AMBER)
    ax.annotate("", xy=(35, 27), xytext=(45, 27), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_AMBER))
    ax.text(40, 29, "Neither", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_AMBER)

    save_fig(fig, "fig05_verdict_decision_tree.png")


# =============================================================================
# 6. UML Sequence Diagram (Fig 06)
# =============================================================================
def generate_fig06():
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 95)
    ax.axis("off")

    ax.text(70, 91, "HALLUCIGUARD END-TO-END VERIFICATION SEQUENCE", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(70, 87.5, "UML 2.5 Sequence Diagram: Inter-Agent Bus and Subsystem Lifelines",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    # Lifelines
    actors = [
        ("User / API", 15),
        ("Base LLM Service", 35),
        ("Detector Agent", 55),
        ("Verifier Pipeline", 78),
        ("n8n Retrieval Service", 102),
        ("Evidence & NLI Engine", 125),
    ]

    for name, x in actors:
        rect = patches.FancyBboxPatch((x - 9, 78), 18, 6, boxstyle="round,pad=0.3",
                                      facecolor=C_CARD_BG, edgecolor=C_NAVY, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 81, name, ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_NAVY)
        ax.plot([x, x], [10, 78], color=C_CARD_BORDER, linestyle="--", lw=1.2)

    # Sequence Messages
    seq = [
        (15, 35, 73, "1: POST /verify (query)", C_BLUE),
        (35, 35, 68, "2: OpenRouter API call (Qwen3)", C_TEXT_MUTED),
        (35, 55, 63, "3: detect(query, draft_response)", C_NAVY),
        (55, 55, 58, "4: DistilBERT inference (prob, risk_level)", C_TEXT_MUTED),
        (55, 78, 53, "5: verify(claims) [if risk ≥ MED]", C_TEAL),
        (78, 102, 48, "6: retrieve_evidence(claims, domain)", C_PURPLE),
        (102, 102, 43, "7: Query primary sources + Tavily fallback", C_TEXT_MUTED),
        (102, 78, 38, "8: Return normalized passages + trace", C_PURPLE),
        (78, 125, 33, "9: rerank(BGE) + nli_classify(DeBERTa)", C_CYAN),
        (125, 125, 28, "10: Compute support, contra, trust, verdict", C_TEXT_MUTED),
        (125, 78, 23, "11: ClaimReport & Citation Items", C_CYAN),
        (78, 15, 18, "12: VerifierOutputV2 with 4-State Verdict", C_GREEN),
    ]

    for x1, x2, y, msg, col in seq:
        if x1 == x2:
            # Self call
            ax.annotate("", xy=(x1 + 6, y - 2), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=col))
            ax.plot([x1, x1 + 6, x1 + 6, x1], [y, y, y - 2, y - 2], color=col, lw=1.2)
            ax.text(x1 + 7.5, y - 1, msg, ha="left", va="center", fontsize=7.2, color=col)
        else:
            ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=col))
            ax.text((x1 + x2)/2, y + 1.2, msg, ha="center", va="center", fontsize=7.5, fontweight="bold", color=col)

    save_fig(fig, "fig06_uml_sequence.png")


# =============================================================================
# 7. Deployment Architecture (Fig 07)
# =============================================================================
def generate_fig07():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(70, 86, "HALLUCIGUARD PHYSICAL & LOGICAL DEPLOYMENT TOPOLOGY", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(70, 82.5, "Hybrid Edge-to-Cloud Deployment: Local GPU Inference Engine + n8n Cloud + OpenRouter",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    # Node 1: Client Layer
    rect = patches.FancyBboxPatch((5, 15), 25, 62, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor=C_CARD_BG, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(17.5, 72, "CLIENT LAYER", ha="center", va="center", fontsize=11, fontweight="bold", color=C_BLUE)
    ax.text(17.5, 55, "Next.js 15 Web Frontend\nTailwind CSS Dashboard\n\nFastAPI REST Clients\nPython SDK / CLI Tools\n\nInspectors & Audit UI",
            ha="center", va="center", fontsize=8.5, color=C_TEXT_DARK)

    # Node 2: Local Python GPU Host
    rect = patches.FancyBboxPatch((35, 10), 60, 68, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor="#F0FDF4", edgecolor=C_GREEN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(65, 73, "HALLUCIGUARD PYTHON RUNTIME (GPU HOST)", ha="center", va="center", fontsize=11, fontweight="bold", color=C_GREEN)
    ax.text(65, 68, "Windows 11 / Linux (Python 3.11 - 3.13) • CUDA Accelerated", ha="center", va="center", fontsize=8, color=C_TEXT_MUTED)

    # Inner boxes
    rect1 = patches.FancyBboxPatch((38, 48), 54, 16, boxstyle="round,pad=0.4", facecolor=C_BG, edgecolor=C_CARD_BORDER)
    ax.add_patch(rect1)
    ax.text(65, 59, "FastAPI Service Orchestrator (Port 8002 / 8000)", ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(65, 52, "• /verify, /health, /detect endpoints\n• BaseLLMDetectorVerifierService • SQLite Verification Cache", ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    rect2 = patches.FancyBboxPatch((38, 14), 54, 30, boxstyle="round,pad=0.4", facecolor=C_BG, edgecolor=C_CARD_BORDER)
    ax.add_patch(rect2)
    ax.text(65, 39, "Local PyTorch Model Pipeline", ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(65, 27, "1. HaluEval DistilBERT (SequenceClassifier)\n2. BAAI/bge-reranker-large (CrossEncoder)\n3. cross-encoder/nli-deberta-v3-base (NLI Engine)\n4. RelationVerifier & SVO Triple Extractor\n5. Calibrated EvidenceScorer & Verdict Engine",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    # Node 3: External Cloud Services
    rect = patches.FancyBboxPatch((100, 15), 35, 62, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor="#FDF4FF", edgecolor=C_PURPLE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(117.5, 72, "EXTERNAL CLOUD SERVICES", ha="center", va="center", fontsize=11, fontweight="bold", color=C_PURPLE)
    ax.text(117.5, 54, "1. OpenRouter API\n   • Qwen/Qwen3-4B & 14B\n\n2. n8n Cloud Workspace\n   • Webhook /halluciguard-verify-v2\n   • Authoritative Source Adapters\n\n3. Tavily AI Search API\n   • Multi-query Web Fallback\n   • Deep Markdown Extraction",
            ha="center", va="center", fontsize=8.2, color=C_TEXT_DARK)

    # Connectors
    ax.annotate("", xy=(35, 50), xytext=(30, 50), arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color=C_BLUE))
    ax.annotate("", xy=(100, 50), xytext=(95, 50), arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color=C_PURPLE))

    save_fig(fig, "fig07_deployment_architecture.png")


def generate_all_diagrams():
    print("Generating High-Resolution Architectural Diagrams...")
    generate_fig01()
    generate_fig02()
    generate_fig03()
    generate_fig04()
    generate_fig05()
    generate_fig06()
    generate_fig07()
    print("All diagrams generated successfully in docs/diagrams/\n")


if __name__ == "__main__":
    generate_all_diagrams()
