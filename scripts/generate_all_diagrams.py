"""
HalluciGuard Master Diagram Generator (All 15 Figures).
Generates publication-quality, FAANG-level light-theme technical figures.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_NAVY = "#1A365D"       # Primary Navy
C_BLUE = "#2563EB"       # Accent Blue
C_TEAL = "#0D9488"       # Secondary Teal
C_CYAN = "#0284C7"       # Accent Cyan
C_BG = "#FFFFFF"         # Pure White
C_CARD_BG = "#F8FAFC"    # Pale Ice / Slate
C_CARD_BORDER = "#CBD5E1"# Subtle Border
C_TEXT_DARK = "#0F172A"  # Charcoal / Black
C_TEXT_MUTED = "#475569" # Slate Muted
C_GREEN = "#16A34A"      # Success / Verified
C_RED = "#DC2626"        # Danger / Contradicted
C_AMBER = "#D97706"      # Warning / Unverified
C_PURPLE = "#7C3AED"     # Special / Conflicted

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def save_fig(fig, filename: str):
    path = DIAGRAMS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  [+] Saved diagram: {path.name}")


# 1. System Architecture
def generate_fig01():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 90)
    ax.axis("off")

    ax.text(70, 86, "HALLUCIGUARD — SYSTEM-LEVEL ARCHITECTURE", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_NAVY)
    ax.text(70, 82.5, "End-to-End Pipeline: OpenRouter LLM -> HaluEval Detector -> n8n Retrieval -> BGE Reranker -> DeBERTa NLI -> 4-State Verdict",
            ha="center", va="center", fontsize=9.5, color=C_TEXT_MUTED)

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

    draw_card(5, 62, 24, 15, "1. Base LLM Service", "OpenRouter / Qwen3-4B\n(Candidate Generation)", C_CARD_BG, C_BLUE, "OPENROUTER")
    ax.text(17, 58, "User Prompt", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_BLUE)
    draw_arrow(17, 62, 17, 54)

    draw_card(5, 38, 24, 16, "2. Detector Agent", "HaluEval DistilBERT\nBinary Classification\n[prob, conf, risk_tier]", C_CARD_BG, C_TEAL, "LOCAL INFERENCE")
    draw_card(36, 42, 22, 12, "3. Risk Gate Router", "LOW (<=0.30) -> Accept\nMED/HIGH (>=0.50) -> Verify", C_CARD_BG, C_AMBER, "ROUTING GATE")
    draw_arrow(29, 46, 36, 46, "draft response")

    draw_card(36, 18, 22, 12, "Accept Path", "Direct Pass-Through\nLatency < 50ms", "#ECFDF5", C_GREEN, "PASSED")
    draw_arrow(47, 42, 47, 30, "LOW Risk", C_GREEN)

    draw_card(65, 58, 26, 16, "4. Claim Decomposer", "Proposition Extraction\nPronoun Resolution\nSVO Triples & Entities", C_CARD_BG, C_CYAN, "VERIFIER AGENT")
    draw_arrow(58, 48, 65, 62, "MED/HIGH Risk", C_RED)

    draw_card(98, 58, 36, 18, "5. n8n Retrieval V2", "Domain Switch (5 Domains)\nPrimary: Wikipedia, PubMed, NVD, arXiv, SEC\nQuality Gate & Tavily Fallback", C_CARD_BG, C_PURPLE, "n8n CLOUD")
    draw_arrow(91, 66, 98, 66, "normalized claims")

    draw_card(98, 32, 36, 16, "6. BGE Reranker Large", "BAAI/bge-reranker-large\nCross-Encoder Semantic Relevance\nIndependent bge_score in [0, 1]", C_CARD_BG, C_BLUE, "CUDA / PyTorch")
    draw_arrow(116, 58, 116, 48, "raw passages")

    draw_card(65, 32, 26, 16, "7. DeBERTa-v3 NLI", "cross-encoder/nli-deberta-v3\nPremise=Evidence, Hyp=Claim\n[Entailment, Contradiction, Neutral]", C_CARD_BG, C_TEAL, "CUDA / PyTorch")
    draw_arrow(98, 40, 91, 40, "ranked evidence")

    draw_card(65, 8, 26, 16, "8. Evidence Scorer", "Relation Check + Authority Weight\n4-Class: Supp/Contra/Neut/Irrel\nCalibrated Trust & Confidence", C_CARD_BG, C_NAVY, "DECISION ENGINE")
    draw_arrow(78, 32, 78, 24, "NLI triples")

    draw_card(98, 8, 36, 16, "9. Final 4-State Verdict", "VERIFIED | CONTRADICTED\nUNVERIFIED | CONFLICTED\nTrace Provenance & Explanations", "#EFF6FF", C_NAVY, "DECISION OUTPUT")
    draw_arrow(91, 16, 98, 16, "scored claim reports")

    save_fig(fig, "fig01_system_architecture.png")


# 2. Detector Architecture & Risk Routing
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

    rect = patches.FancyBboxPatch((5, 45), 28, 20, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(19, 61, "DetectionInput", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(19, 53, "• user_query: str\n• llm_response: str\n• max_length: 384 tokens\n• Format: [CLS] Q [SEP] R [SEP]",
            ha="center", va="center", fontsize=8, color=C_TEXT_MUTED)

    rect = patches.FancyBboxPatch((40, 42), 34, 26, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_TEAL, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(57, 64, "HaluEval DistilBERT Engine", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(57, 53, "Model: Manjunath2000006/halluciguard-detector\nBase: distilbert-base-uncased\nForward Pass: Logits -> Softmax\n\nExecution Diagnostics:\n• detector_model_loaded: bool\n• detector_inference_executed: bool\n• detector_degraded: bool\n• detector_model_source: str",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    ax.annotate("", xy=(40, 55), xytext=(33, 55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    rect = patches.FancyBboxPatch((82, 42), 43, 26, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor=C_CARD_BG, edgecolor=C_CYAN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(103.5, 64, "DetectionResult Schema", ha="center", va="center", fontsize=10, fontweight="bold", color=C_NAVY)
    ax.text(103.5, 52, "• confidence_score: float in [0.0, 1.0]\n• hallucination_probability: float in [0.0, 1.0]\n• risk_level: RiskLevel (LOW / MED / HIGH)\n• next_action: NextAction (Accept / Verify)\n• model_source: 'halueval-distilbert'\n• diagnostics (provenance + degraded flags)",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    ax.annotate("", xy=(82, 55), xytext=(74, 55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    ax.text(65, 34, "THREE-TIER RISK ROUTING POLICY", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NAVY)

    rect = patches.FancyBboxPatch((8, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#ECFDF5", edgecolor=C_GREEN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(25, 24, "LOW RISK (<= 0.30)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_GREEN)
    ax.text(25, 16, "Action: ACCEPT\n• Verifier Skipped\n• High factuality confidence\n• Output directly returned", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    rect = patches.FancyBboxPatch((48, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#FFFBEB", edgecolor=C_AMBER, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(65, 24, "MEDIUM RISK (0.30 - 0.50)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_AMBER)
    ax.text(65, 16, "Action: VERIFY\n• Verifier Triggered\n• Potential factual uncertainty\n• Full evidence grounding", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    rect = patches.FancyBboxPatch((88, 10), 34, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                                  facecolor="#FEF2F2", edgecolor=C_RED, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(105, 24, "HIGH RISK (>= 0.50)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_RED)
    ax.text(105, 16, "Action: VERIFY\n• Verifier Mandatory\n• Elevated hallucination probability\n• Multi-source cross-checking", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    save_fig(fig, "fig02_detector_routing.png")


# 3. n8n Workflow Topology & Domain Routing
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

    draw_node(4, 52, 22, 14, "Webhook: Receive Claim", "POST /halluciguard-verify-v2\nX-API-Key Authentication\nExtracts claim & mode", C_CARD_BG, C_BLUE)
    draw_node(30, 52, 22, 14, "LLM: Analyze Claim", "OpenRouter / Qwen3-14B\nExtracts Queries & Entities\nClassifies Routing Domain", C_CARD_BG, C_TEAL)
    draw_node(56, 52, 20, 14, "Code: Build Context", "Builds Runtime Payload\nNormalizes Query Tokens\nSets Timeouts & Flags", C_CARD_BG, C_CYAN)

    ax.annotate("", xy=(30, 59), xytext=(26, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))
    ax.annotate("", xy=(56, 59), xytext=(52, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    draw_node(80, 52, 18, 14, "Domain Switch", "Rule-Based Router\n5 Domain Ports\n+ Fallback Port", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(80, 59), xytext=(76, 59), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

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

    draw_node(74, 18, 24, 13, "Assemble Primary", "Extracts text snippets\nComputes token score\nAttaches provenance", C_CARD_BG, C_NAVY)
    for _, _, dy in domains:
        ax.annotate("", xy=(86, 31), xytext=(104, dy + 5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_CARD_BORDER))

    draw_node(46, 18, 24, 13, "Quality Evaluation", "usable_count >= 2 ?\nrelevant_count >= 2 ?\nDetermines Tavily Need", "#FEF3C7", C_AMBER)
    ax.annotate("", xy=(70, 24.5), xytext=(74, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    draw_node(18, 18, 24, 13, "Tavily Web Fallback", "Advanced Search (multi-query)\nBasic Markdown Extract\nExtract Latency & Scores", "#EFF6FF", C_BLUE)
    ax.annotate("", xy=(42, 24.5), xytext=(46, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_AMBER))

    draw_node(4, 18, 12, 24, "Response", "Merge & Dedup\nTrace Object\nJSON Return", "#ECFDF5", C_GREEN)
    ax.annotate("", xy=(16, 24.5), xytext=(18, 24.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    save_fig(fig, "fig03_n8n_retrieval_topology.png")


# 4. Evidence Lifecycle State Machine
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

    draw_state(5, 36, 25, 18, "1. Ingested Passage", "• Raw text snippet\n• Source provenance URL\n• adapter_score hint\n• relevance_score = 0.0", C_CARD_BG, C_BLUE)
    draw_state(38, 36, 25, 18, "2. Relevance Gating", "• BGE Semantic Score\n• Gate Threshold: 0.20\n• If score < 0.20 ->\n  Marked IRRELEVANT", C_CARD_BG, C_CYAN)
    ax.annotate("", xy=(38, 45), xytext=(30, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    draw_state(71, 36, 25, 18, "3. NLI Classification", "• DeBERTa-v3-base forward\n• P(Entail), P(Contra), P(Neut)\n• SVO Relation Check Override\n• Non-assertive myth filter", C_CARD_BG, C_TEAL)
    ax.annotate("", xy=(71, 45), xytext=(63, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    draw_state(104, 52, 22, 12, "SUPPORTING", "Entailment >= 0.35\n& Entail > Contra", "#ECFDF5", C_GREEN)
    draw_state(104, 36, 22, 12, "CONTRADICTING", "Contra >= 0.35\n& Contra > Entail", "#FEF2F2", C_RED)
    draw_state(104, 20, 22, 12, "NEUTRAL", "Informational context\nNo decisive polarity", "#FFFBEB", C_AMBER)
    draw_state(104, 4, 22, 12, "IRRELEVANT", "BGE score < 0.20\nExcluded from verdict", "#F1F5F9", C_CARD_BORDER)

    ax.annotate("", xy=(104, 58), xytext=(96, 48), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_GREEN))
    ax.annotate("", xy=(104, 42), xytext=(96, 45), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_RED))
    ax.annotate("", xy=(104, 26), xytext=(96, 42), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_AMBER))
    ax.annotate("", xy=(104, 10), xytext=(63, 39), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_CARD_BORDER))

    save_fig(fig, "fig04_evidence_lifecycle.png")


# 5. Four-State Verdict Decision Tree
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

    draw_box(45, 60, 40, 14, "Evidence Scoring Inputs", "support_score = 0.70·max + 0.30·avg + bonus\ncontra_score = 0.70·max + 0.30·avg + bonus\nEffective Weight = Rel · Cred · Recency", C_CARD_BG, C_BLUE)
    draw_box(45, 40, 40, 14, "Conflict Evaluation", "support_score >= 0.30 AND\ncontra_score >= 0.30 AND\n|support - contra| < 0.15 ?", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(65, 54), xytext=(65, 60), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    draw_box(95, 40, 30, 14, "CONFLICTED", "Credible conflicting evidence\nfrom authoritative sources\nE.g., active scientific debate", "#F5F3FF", C_PURPLE)
    ax.annotate("", xy=(95, 47), xytext=(85, 47), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_PURPLE))
    ax.text(90, 49, "YES", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_PURPLE)

    draw_box(45, 20, 40, 14, "Polarity Dominance Check", "support_score >= 0.30 & supp > contra ?\nOR\ncontra_score >= 0.25 & contra > supp ?", C_CARD_BG, C_CYAN)
    ax.annotate("", xy=(65, 34), xytext=(65, 40), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))
    ax.text(68, 37, "NO", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_NAVY)

    draw_box(95, 22, 30, 12, "VERIFIED", "support_score >= 0.30\nHigh evidence grounding\nZero contradiction", "#ECFDF5", C_GREEN)
    ax.annotate("", xy=(95, 28), xytext=(85, 28), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_GREEN))
    ax.text(90, 30, "Support", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_GREEN)

    draw_box(95, 6, 30, 12, "CONTRADICTED", "contra_score >= 0.25\nExplicit refutation found\nOr relation mismatch", "#FEF2F2", C_RED)
    ax.annotate("", xy=(95, 12), xytext=(85, 24), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_RED))
    ax.text(90, 18, "Contra", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_RED)

    draw_box(5, 20, 30, 14, "UNVERIFIED", "Safe Abstention\nInsufficient evidence / low relevance\nRefuses to hallucinate truth", "#FFFBEB", C_AMBER)
    ax.annotate("", xy=(35, 27), xytext=(45, 27), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_AMBER))
    ax.text(40, 29, "Neither", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_AMBER)

    save_fig(fig, "fig05_verdict_decision_tree.png")


# 6. UML Sequence Diagram
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

    seq = [
        (15, 35, 73, "1: POST /verify (query)", C_BLUE),
        (35, 35, 68, "2: OpenRouter API call (Qwen3)", C_TEXT_MUTED),
        (35, 55, 63, "3: detect(query, draft_response)", C_NAVY),
        (55, 55, 58, "4: DistilBERT inference (prob, risk_level)", C_TEXT_MUTED),
        (55, 78, 53, "5: verify(claims) [if risk >= MED]", C_TEAL),
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
            ax.annotate("", xy=(x1 + 6, y - 2), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=col))
            ax.plot([x1, x1 + 6, x1 + 6, x1], [y, y, y - 2, y - 2], color=col, lw=1.2)
            ax.text(x1 + 7.5, y - 1, msg, ha="left", va="center", fontsize=7.2, color=col)
        else:
            ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=col))
            ax.text((x1 + x2)/2, y + 1.2, msg, ha="center", va="center", fontsize=7.5, fontweight="bold", color=col)

    save_fig(fig, "fig06_uml_sequence.png")


# 7. Deployment Architecture
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

    rect = patches.FancyBboxPatch((5, 15), 25, 62, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor=C_CARD_BG, edgecolor=C_BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(17.5, 72, "CLIENT LAYER", ha="center", va="center", fontsize=11, fontweight="bold", color=C_BLUE)
    ax.text(17.5, 55, "Next.js 15 Web Frontend\nTailwind CSS Dashboard\n\nFastAPI REST Clients\nPython SDK / CLI Tools\n\nInspectors & Audit UI",
            ha="center", va="center", fontsize=8.5, color=C_TEXT_DARK)

    rect = patches.FancyBboxPatch((35, 10), 60, 68, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor="#F0FDF4", edgecolor=C_GREEN, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(65, 73, "HALLUCIGUARD PYTHON RUNTIME (GPU HOST)", ha="center", va="center", fontsize=11, fontweight="bold", color=C_GREEN)
    ax.text(65, 68, "Windows 11 / Linux (Python 3.11 - 3.13) • CUDA Accelerated", ha="center", va="center", fontsize=8, color=C_TEXT_MUTED)

    rect1 = patches.FancyBboxPatch((38, 48), 54, 16, boxstyle="round,pad=0.4", facecolor=C_BG, edgecolor=C_CARD_BORDER)
    ax.add_patch(rect1)
    ax.text(65, 59, "FastAPI Service Orchestrator (Port 8002 / 8000)", ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(65, 52, "• /verify, /health, /detect endpoints\n• BaseLLMDetectorVerifierService • SQLite Verification Cache", ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    rect2 = patches.FancyBboxPatch((38, 14), 54, 30, boxstyle="round,pad=0.4", facecolor=C_BG, edgecolor=C_CARD_BORDER)
    ax.add_patch(rect2)
    ax.text(65, 39, "Local PyTorch Model Pipeline", ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(65, 27, "1. HaluEval DistilBERT (SequenceClassifier)\n2. BAAI/bge-reranker-large (CrossEncoder)\n3. cross-encoder/nli-deberta-v3-base (NLI Engine)\n4. RelationVerifier & SVO Triple Extractor\n5. Calibrated EvidenceScorer & Verdict Engine",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    rect = patches.FancyBboxPatch((100, 15), 35, 62, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor="#FDF4FF", edgecolor=C_PURPLE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(117.5, 72, "EXTERNAL CLOUD SERVICES", ha="center", va="center", fontsize=11, fontweight="bold", color=C_PURPLE)
    ax.text(117.5, 54, "1. OpenRouter API\n   • Qwen/Qwen3-4B & 14B\n\n2. n8n Cloud Workspace\n   • Webhook /halluciguard-verify-v2\n   • Authoritative Source Adapters\n\n3. Tavily AI Search API\n   • Multi-query Web Fallback\n   • Deep Markdown Extraction",
            ha="center", va="center", fontsize=8.2, color=C_TEXT_DARK)

    ax.annotate("", xy=(35, 50), xytext=(30, 50), arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color=C_BLUE))
    ax.annotate("", xy=(100, 50), xytext=(95, 50), arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color=C_PURPLE))

    save_fig(fig, "fig07_deployment_architecture.png")


# 8. UML Component Diagram
def generate_fig08():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "UML COMPONENT DIAGRAM — AGENTS & SUBSYSTEM INTERCONNECTS", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Modular Boundaries, Explicit I/O Contracts & Shared Inter-Agent State Bus",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_comp(x, y, w, h, name, interfaces, color=C_CARD_BG, border=C_NAVY):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, f"<<component>>\n{name}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 1.5, interfaces, ha="center", va="center", fontsize=7.2, color=C_TEXT_DARK, multialignment="center")

    draw_comp(5, 42, 34, 22, "BaseLLMService", "Provides:\n• generate()\n• health_check()\nRequires:\n• OpenRouter REST API")
    draw_comp(48, 42, 34, 22, "DetectorAgent", "Provides:\n• detect()\n• assess_risk()\nRequires:\n• HaluEval Weights\n• PyTorch runtime")
    draw_comp(91, 42, 34, 22, "N8NRetrievalClient", "Provides:\n• retrieve_evidence()\n• health_check()\nRequires:\n• n8n Webhook V2")

    draw_comp(5, 10, 34, 24, "VerificationPipeline", "Provides:\n• verify()\n• health_check()\nRequires:\n• BGE Reranker\n• DeBERTa NLI\n• EvidenceScorer")
    draw_comp(48, 10, 34, 24, "Orchestration Supervisor", "Provides:\n• StateGraph.ainvoke()\n• run_verification()\nRequires:\n• Inter-agent bus\n• Audit trace logger")
    draw_comp(91, 10, 34, 24, "Memory & Cache", "Provides:\n• SQLite VerificationCache\n• EpisodicMemoryStore\nRequires:\n• SQLite driver (aiosqlite)")

    # Connectors
    ax.plot([39, 48], [53, 53], color=C_BLUE, lw=1.5)
    ax.plot([82, 91], [53, 53], color=C_TEAL, lw=1.5)
    ax.plot([22, 22], [42, 34], color=C_NAVY, lw=1.5)
    ax.plot([65, 65], [42, 34], color=C_PURPLE, lw=1.5)
    ax.plot([39, 48], [22, 22], color=C_GREEN, lw=1.5)
    ax.plot([82, 91], [22, 22], color=C_AMBER, lw=1.5)

    save_fig(fig, "fig08_uml_component.png")


# 9. UML Activity Flow with Certification Mode
def generate_fig09():
    fig, ax = plt.subplots(figsize=(13, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 85)
    ax.axis("off")

    ax.text(65, 81, "UML ACTIVITY FLOW: RESILIENT PRODUCTION VS FAIL-CLOSED CERTIFICATION", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color=C_NAVY)
    ax.text(65, 77.5, "Branching Logic: Default Fail-Soft Recovery vs Strict Controlled Failures (CERTIFICATION_MODE=true)",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_act(x, y, w, h, text, color=C_CARD_BG, border=C_BLUE):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold", color=C_NAVY, multialignment="center")

    draw_act(48, 64, 34, 8, "Start: Receive Claim Verification Request")
    draw_act(48, 50, 34, 8, "Execute Detector Inference\n(HaluEval DistilBERT)")
    ax.annotate("", xy=(65, 58), xytext=(65, 64), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # Decision: Cert mode?
    rect = patches.Polygon([[65, 44], [78, 38], [65, 32], [52, 38]], facecolor="#FEF3C7", edgecolor=C_AMBER, lw=1.3)
    ax.add_patch(rect)
    ax.text(65, 38, "Model Degraded\nOR Mock Evidence?", ha="center", va="center", fontsize=7, fontweight="bold", color=C_NAVY)
    ax.annotate("", xy=(65, 44), xytext=(65, 50), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # Branch 1: Normal production
    draw_act(8, 20, 38, 14, "Normal Mode (CERT=false)\n• Degraded neutral fallback\n• Informational trace logged\n• Best-effort UNVERIFIED emitted", "#ECFDF5", C_GREEN)
    ax.annotate("", xy=(27, 34), xytext=(52, 38), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_GREEN))
    ax.text(38, 40, "CERT=false", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_GREEN)

    # Branch 2: Certification mode
    draw_act(84, 20, 38, 14, "Certification Mode (CERT=true)\n• Raise CertificationError\n• Refuses to guess or fake verdict\n• Emits Controlled Failure Block", "#FEF2F2", C_RED)
    ax.annotate("", xy=(103, 34), xytext=(78, 38), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_RED))
    ax.text(92, 40, "CERT=true", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_RED)

    # End
    draw_act(48, 4, 34, 8, "End: Emit Verifier Output / Failure Trace", C_CARD_BG, C_NAVY)
    ax.annotate("", xy=(65, 12), xytext=(27, 20), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_CARD_BORDER))
    ax.annotate("", xy=(65, 12), xytext=(103, 20), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_CARD_BORDER))

    save_fig(fig, "fig09_uml_activity.png")


# 10. Module Dependency Circuit
def generate_fig10():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "HALLUCIGUARD MODULE DEPENDENCY CIRCUIT", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Strict Layered Dependency Architecture: Services -> Agents -> Pipeline -> Engine",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_mod(x, y, w, h, name, path, color=C_CARD_BG, border=C_BLUE):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                      facecolor=color, edgecolor=border, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 1.2, path, ha="center", va="center", fontsize=7, color=C_TEXT_MUTED)

    draw_mod(5, 52, 35, 14, "services.base_llm_service", "BaseLLMService (OpenRouter)", C_CARD_BG, C_BLUE)
    draw_mod(48, 52, 35, 14, "services.llm_detector_verifier", "BaseLLMDetectorVerifierService", C_CARD_BG, C_TEAL)
    draw_mod(90, 52, 35, 14, "agents.detector_agent", "DetectorAgent & HaluEval", C_CARD_BG, C_CYAN)

    draw_mod(5, 28, 35, 14, "agents.verifier_agent.claims", "ClaimDecomposer & Normalizer", C_CARD_BG, C_PURPLE)
    draw_mod(48, 28, 35, 14, "agents.verifier_agent.api.pipeline", "VerificationPipeline Orchestrator", C_CARD_BG, C_GREEN)
    draw_mod(90, 28, 35, 14, "services.n8n_retrieval_client", "N8NRetrievalClient (Webhook V2)", C_CARD_BG, C_AMBER)

    draw_mod(5, 6, 35, 14, "rerankers.cross_encoder", "BAAI/bge-reranker-large", C_CARD_BG, C_NAVY)
    draw_mod(48, 6, 35, 14, "nli.entailment", "cross-encoder/nli-deberta-v3", C_CARD_BG, C_NAVY)
    draw_mod(90, 6, 35, 14, "scorers.evidence_scorer", "EvidenceScorer & RelationVerifier", C_CARD_BG, C_NAVY)

    # Connections
    ax.annotate("", xy=(48, 59), xytext=(40, 59), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(90, 59), xytext=(83, 59), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(65, 42), xytext=(65, 52), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(40, 35), xytext=(48, 35), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(83, 35), xytext=(90, 35), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(22, 20), xytext=(48, 28), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(65, 20), xytext=(65, 28), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))
    ax.annotate("", xy=(107, 20), xytext=(83, 28), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_NAVY))

    save_fig(fig, "fig10_module_circuit.png")


# 11. Data Flow Diagram (Level-0 & Level-1 DFD)
def generate_fig11():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "LEVEL-0 & LEVEL-1 DATA FLOW DIAGRAM (DFD)", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Data Transformations: Raw Query -> LLM Draft -> Risk Metric -> Normalized Triples -> Verdict",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_dfd_node(x, y, w, h, text, is_process=True):
        if is_process:
            rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.0",
                                          facecolor=C_CARD_BG, edgecolor=C_TEAL, linewidth=1.4)
        else:
            rect = patches.Rectangle((x, y), w, h, facecolor="#F1F5F9", edgecolor=C_NAVY, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold", color=C_NAVY, multialignment="center")

    draw_dfd_node(5, 34, 18, 14, "External\nUser", is_process=False)
    draw_dfd_node(30, 34, 22, 14, "P1.0\nGenerate &\nDetect Risk")
    draw_dfd_node(60, 34, 22, 14, "P2.0\nDecompose &\nRetrieve Evidence")
    draw_dfd_node(90, 34, 22, 14, "P3.0\nSemantic Rerank\n& NLI Inference")
    draw_dfd_node(116, 34, 12, 14, "P4.0\nScore &\nVerdict")

    # Data store
    rect = patches.Rectangle((55, 6), 32, 12, facecolor="#EFF6FF", edgecolor=C_BLUE, linewidth=1.3)
    ax.add_patch(rect)
    ax.text(71, 12, "D1: Verification Cache & Trace Store\n(SQLite + Episodic DB)", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_BLUE)

    # Arrows
    ax.annotate("", xy=(30, 41), xytext=(23, 41), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))
    ax.text(26.5, 43, "query", ha="center", va="center", fontsize=7.2, color=C_NAVY)

    ax.annotate("", xy=(60, 41), xytext=(52, 41), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))
    ax.text(56, 43, "draft + risk", ha="center", va="center", fontsize=7.2, color=C_NAVY)

    ax.annotate("", xy=(90, 41), xytext=(82, 41), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))
    ax.text(86, 43, "passages", ha="center", va="center", fontsize=7.2, color=C_NAVY)

    ax.annotate("", xy=(116, 41), xytext=(112, 41), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_NAVY))

    # Return
    ax.annotate("", xy=(14, 34), xytext=(122, 34), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=C_GREEN, connectionstyle="arc3,rad=-0.3"))
    ax.text(68, 22, "Structured VerifierOutputV2 with Citations & 4-State Verdict", ha="center", va="center", fontsize=8, fontweight="bold", color=C_GREEN)

    save_fig(fig, "fig11_data_flow_diagram.png")


# 12. Relation Verification & SVO Triples
def generate_fig12():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 70)
    ax.axis("off")

    ax.text(65, 66, "RELATION VERIFIER & SVO TRIPLES EXTRACTION ENGINE", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 62.5, "Structural Fact Extraction: Subject-Verb-Object Triples & Entity Alignment",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_card(x, y, w, h, title, desc, color=C_CARD_BG, border=C_BLUE):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 1.0, desc, ha="center", va="center", fontsize=7.5, color=C_TEXT_DARK, multialignment="center")

    draw_card(5, 34, 35, 22, "Claim Extraction", "Claim: 'Allu Arjun father is Chiranjeevi'\n\nExtracted SVO Triple:\n• Subject: Allu Arjun\n• Relation: father / parent\n• Object: Chiranjeevi", C_CARD_BG, C_RED)
    draw_card(48, 34, 35, 22, "Evidence Extraction", "Passage: 'Allu Arjun is son of Allu Aravind'\n\nExtracted SVO Triple:\n• Subject: Allu Arjun\n• Relation: father / parent\n• Object: Allu Aravind", C_CARD_BG, C_GREEN)
    draw_card(91, 34, 35, 22, "Relation Matcher", "Comparison Result:\n• Status: OBJECT_MISMATCH\n• Claim Obj != Evidence Obj\n• Chiranjeevi != Allu Aravind\n-> Direct Contradiction Override", "#FEF2F2", C_RED)

    ax.annotate("", xy=(48, 45), xytext=(40, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))
    ax.annotate("", xy=(91, 45), xytext=(83, 45), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NAVY))

    # Bottom summary box
    rect = patches.FancyBboxPatch((15, 8), 100, 18, boxstyle="round,pad=0.4", facecolor="#EFF6FF", edgecolor=C_BLUE)
    ax.add_patch(rect)
    ax.text(65, 20, "RELATION VERIFICATION BYPASSES WORD COVERAGE SUPPRESSION", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_BLUE)
    ax.text(65, 13, "When an explicit Subject-Object mismatch is verified (e.g. Chiranjeevi vs Allu Aravind),\nthe scorer assigns contradiction_score >= 0.95, preventing false neutral classifications.", ha="center", va="center", fontsize=8, color=C_TEXT_DARK)

    save_fig(fig, "fig12_relation_verification.png")


# 13. Evidence Scoring Mathematics
def generate_fig13():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "EVIDENCE SCORING FORMULAS & CONFIDENCE CALIBRATION", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Mathematical Grounding: Effective Weight, Polarity Scoring, and Consensus Gating",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_formula(x, y, w, h, title, formula, color=C_CARD_BG, border=C_NAVY):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 1.2, formula, ha="center", va="center", fontsize=7.5, color=C_TEXT_DARK, multialignment="center")

    draw_formula(5, 44, 58, 24, "1. Relevance & Effective Weight",
                 "relevance_weight = max(0.20, min(1.0, score^0.25))\nbase_weight = credibility · recency · relevance · validity\nsupport_signal = entailment · (1.0 - 0.35 · neutral)\ncontra_signal = contradiction · (1.0 - 0.35 · neutral)")

    draw_formula(67, 44, 58, 24, "2. Aggregate Polarity Scores",
                 "support_score = min(1.0, 0.70·max + 0.30·avg + bonus)\ncontra_score = min(1.0, 0.70·max + 0.30·avg + bonus)\nwhere bonus = min(0.15, 0.05 · (distinct_sources - 1))\nDeduplication: max effective weight per canonical URL")

    draw_formula(5, 12, 58, 26, "3. Trust Score Formula",
                 "if support_score > contra_score and support_score >= 0.25:\n  trust_score = support_score · (1.0 - 0.90 · contra_score)\n                + 0.05 · min(1.0, sources / 2.0)\nelse:\n  trust_score = 0.0\nReflects supporting authority reliability.")

    draw_formula(67, 12, 58, 26, "4. Calibrated Confidence Score",
                 "strength = max(support_score, contra_score)\nif strength >= 0.25:\n  consensus = max(0.10, 1.0 - min(support, contra))\n  count_factor = 0.75 + 0.25 · min(1.0, n_passages / 3.0)\n  confidence = strength · consensus · count_factor\nelse:\n  confidence = 0.0")

    save_fig(fig, "fig13_scoring_confidence_math.png")


# 14. Observability Trace Hierarchy
def generate_fig14():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "RUNTIME OBSERVABILITY TRACE & MODEL EXECUTION PROOF", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Comprehensive Provenance Hierarchy: Diagnostics, Gate Signals, and Certification Audit",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    def draw_trace_box(x, y, w, h, title, fields, color=C_CARD_BG, border=C_CYAN):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      facecolor=color, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9, fontweight="bold", color=C_NAVY)
        ax.text(x + w/2, y + h/2 - 1.2, fields, ha="center", va="center", fontsize=7.2, color=C_TEXT_DARK, multialignment="center")

    draw_trace_box(5, 42, 38, 26, "RetrievalTrace Root",
                   "• retrieval_latency_ms: int\n• workflow_version: str ('2.0.0')\n• sources_attempted: list[str]\n• primary_count / fallback_count: int\n• reranker_execution: ModelExecutionTrace\n• nli_execution: ModelExecutionTrace\n• gate_relevance_audit: GateAudit")

    draw_trace_box(46, 42, 38, 26, "ModelExecutionTrace",
                   "• component: 'bge_reranker' | 'deberta_nli'\n• model: 'BAAI/bge-reranker-large'\n• loaded: bool\n• inference_executed: bool\n• degraded: bool\n• device: 'cuda' | 'cpu'\n• latency_ms: float\n• scored_count / batch_size: int")

    draw_trace_box(87, 42, 38, 26, "GateRelevanceAudit",
                   "• gate_time_relevance_signal: float\n• final_bge_relevance_score: float\n• signals_agree: bool\n• min_top_relevance_threshold: float\n• relevance_threshold: float\n• primary_evidence_sufficient: bool")

    # Bottom audit block
    rect = patches.FancyBboxPatch((15, 10), 100, 24, boxstyle="round,pad=0.4", facecolor="#F0FDF4", edgecolor=C_GREEN)
    ax.add_patch(rect)
    ax.text(65, 28, "CERTIFICATION AUDIT PROOF (ANTI-MASQUERADE INVARIANT)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_GREEN)
    ax.text(65, 18, "Every verification response carries cryptographically verifiable proof of model execution:\n1. detector_inference_executed == True (HaluEval DistilBERT forward pass verified)\n2. reranker_execution.inference_executed == True & bge_score != adapter_score\n3. nli_execution.inference_executed == True & status == 'executed' on CUDA device\n4. certification_mode ensures NO mock evidence or degraded fallbacks are silently accepted.",
            ha="center", va="center", fontsize=7.8, color=C_TEXT_DARK)

    save_fig(fig, "fig14_observability_trace.png")


# 15. Threat Model & Adversarial Robustness
def generate_fig15():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(65, 76, "THREAT MODEL & ADVERSARIAL ROBUSTNESS ARCHITECTURE", ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_NAVY)
    ax.text(65, 72.5, "Defense-in-Depth Mitigations Against Prompt Injections, Web Poisoning & False Entailment",
            ha="center", va="center", fontsize=9, color=C_TEXT_MUTED)

    threats = [
        ("Threat 1: Prompt Injection in User Query", "Strict JSON schema validation; system prompts freeze retrieval scope; model outputs are parsed deterministically without executing code."),
        ("Threat 2: Web Search Poisoning / SEO Spam", "Domain authority weighting (e.g. PubMed 0.95 vs blog 0.40); cross-source consensus gating; minimum 2 diverse domains required."),
        ("Threat 3: False Verification via Myth Matching", "Non-assertive claim context filter rejects passages discussing folklores, idioms, hoaxes, nursery rhymes, and debunked myths."),
        ("Threat 4: Subject-Object Substitution Attack", "Structured SVO RelationVerifier extracts named entities and checks object compatibility, forcing CONTRADICTION on mismatches."),
        ("Threat 5: Silent Fallback Masquerade", "Engine-level diagnostics and Fail-Closed Certification Mode raise hard errors on degraded model states rather than guessing."),
    ]

    y_pos = 58
    for title, mitigation in threats:
        rect = patches.FancyBboxPatch((5, y_pos - 9), 120, 10, boxstyle="round,pad=0.3,rounding_size=0.8",
                                      facecolor=C_CARD_BG, edgecolor=C_NAVY, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(8, y_pos - 2.5, title, ha="left", va="center", fontsize=8.2, fontweight="bold", color=C_NAVY)
        ax.text(8, y_pos - 6.5, f"Mitigation: {mitigation}", ha="left", va="center", fontsize=7.2, color=C_TEXT_DARK)
        y_pos -= 12

    save_fig(fig, "fig15_threat_model_mitigation.png")


def generate_all_diagrams():
    print("Generating All 15 High-Resolution Architectural Diagrams...")
    generate_fig01()
    generate_fig02()
    generate_fig03()
    generate_fig04()
    generate_fig05()
    generate_fig06()
    generate_fig07()
    generate_fig08()
    generate_fig09()
    generate_fig10()
    generate_fig11()
    generate_fig12()
    generate_fig13()
    generate_fig14()
    generate_fig15()
    print("Successfully generated all 15 diagrams in docs/diagrams/\n")


if __name__ == "__main__":
    generate_all_diagrams()
