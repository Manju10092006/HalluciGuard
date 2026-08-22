from __future__ import annotations
import time
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.models import VerifierInputV2, VerifierOutputV2, DomainStatistics
from api.pipeline import VerificationPipeline
from adapters.registry import get_registry
from models import get_domain_intelligence_registry, get_model_manager
from metrics import MetricsCollector
from version import get_version_info
from utils.logging import setup_logger, VerificationLogRecord, log_verification_request

logger = setup_logger("api.main")
_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Verifier Agent lifespan...")
    pipeline = VerificationPipeline()
    app.state.pipeline = pipeline
    try:
        await pipeline.cache.init_db()
    except Exception as e:
        logger.warning(f"Failed to initialize SQLite cache: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Verifier Agent...")

app = FastAPI(
    title="HalluciGuard Verifier Agent",
    description="Evidence retrieval and claim verification engine for the HalluciGuard trust layer.",
    version="2.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HalluciGuard Verifier Agent — Live Intelligence Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: #111827;
            --bg-card-hover: #1f2937;
            --border-color: #1f293d;
            --border-accent: #374151;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --verified-green: #10b981;
            --hallucinated-red: #ef4444;
            --mixed-amber: #f59e0b;
            --insufficient-slate: #6b7280;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            line-height: 1.5;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        }

        header {
            border-bottom: 1px solid var(--border-color);
            background: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-badge {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            color: white;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }

        .logo-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--verified-green);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--verified-green);
            box-shadow: 0 0 8px var(--verified-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
            100% { opacity: 1; transform: scale(1); }
        }

        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 2rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-main);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        textarea, select {
            width: 100%;
            background: #0d1322;
            border: 1px solid var(--border-accent);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            font-family: inherit;
            transition: all 0.2s ease;
        }

        textarea:focus, select:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        .sample-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 0.75rem;
        }

        .btn-sample {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-sample:hover {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            border-color: rgba(59, 130, 246, 0.4);
        }

        .btn-primary {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: white;
            border: none;
            padding: 0.9rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .verdict-banner {
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border: 1px solid;
            background: rgba(17, 24, 39, 0.9);
        }

        .verdict-banner.verified {
            border-color: rgba(16, 185, 129, 0.4);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(17, 24, 39, 0.9));
        }

        .verdict-banner.likely_hallucinated {
            border-color: rgba(239, 68, 68, 0.4);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(17, 24, 39, 0.9));
        }

        .verdict-banner.mixed_evidence {
            border-color: rgba(245, 158, 11, 0.4);
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(17, 24, 39, 0.9));
        }

        .verdict-banner.insufficient_evidence {
            border-color: rgba(107, 114, 128, 0.4);
            background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(17, 24, 39, 0.9));
        }

        .verdict-tag {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            text-transform: uppercase;
        }

        .verified .verdict-tag { color: var(--verified-green); }
        .likely_hallucinated .verdict-tag { color: var(--hallucinated-red); }
        .mixed_evidence .verdict-tag { color: var(--mixed-amber); }
        .insufficient_evidence .verdict-tag { color: var(--insufficient-slate); }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }

        .metric-box {
            background: #0d1322;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 4px;
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .runtime-info {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            background: rgba(13, 19, 34, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.85rem;
        }

        .runtime-item span {
            color: var(--text-muted);
            display: block;
            font-size: 0.75rem;
        }

        .runtime-item strong {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        .evidence-card {
            background: #0d1322;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease;
        }

        .evidence-card:hover {
            border-color: var(--border-accent);
        }

        .evidence-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .evidence-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--accent-blue);
            text-decoration: none;
        }

        .evidence-title:hover {
            text-decoration: underline;
        }

        .entailment-badge {
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-entailment { background: rgba(16, 185, 129, 0.15); color: var(--verified-green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-contradiction { background: rgba(239, 68, 68, 0.15); color: var(--hallucinated-red); border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-neutral { background: rgba(107, 114, 128, 0.15); color: var(--insufficient-slate); border: 1px solid rgba(107, 114, 128, 0.3); }

        .evidence-snippet {
            font-size: 0.9rem;
            color: #d1d5db;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }

        .evidence-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.78rem;
            color: var(--text-dim);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.5rem;
        }

        .explanation-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1.25rem;
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--text-main);
        }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-dim);
        }

        .empty-state svg {
            width: 48px;
            height: 48px;
            margin-bottom: 1rem;
            opacity: 0.4;
        }

        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border-left-color: white;
            animation: spin 1s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <div class="logo-group">
                <div class="logo-badge">H</div>
                <div>
                    <div class="logo-title">HalluciGuard Verifier Agent</div>
                    <div style="font-size: 0.75rem; color: var(--text-dim);">Enterprise Multi-Source AI Fact-Checking Engine</div>
                </div>
            </div>
            <div class="status-pill">
                <div class="pulse-dot"></div>
                <span>9-Stage Pipeline Active</span>
            </div>
        </div>
    </header>

    <main>
        <!-- Left Column: Input Form -->
        <section>
            <div class="card">
                <div class="card-header">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                    Verification Request
                </div>

                <form id="verifyForm">
                    <div class="form-group">
                        <label for="domainSelect">Target Knowledge Domain</label>
                        <select id="domainSelect">
                            <option value="healthcare">Healthcare & Medicine (PubMed, OpenFDA)</option>
                            <option value="cybersecurity">Cybersecurity (NVD CVE, MITRE ATT&CK)</option>
                            <option value="finance">Finance & Markets (SEC EDGAR, World Bank)</option>
                            <option value="ai_research">AI & Computer Science (arXiv, Crossref)</option>
                            <option value="legal_general">Legal & Statutes (CourtListener, Wikipedia)</option>
                            <option value="general">General Knowledge (Wikipedia REST)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="claimText">Claim Text to Verify</label>
                        <textarea id="claimText" placeholder="Enter an LLM assertion or claim to verify against authoritative sources..."></textarea>
                        
                        <div class="sample-buttons">
                            <span style="font-size: 0.75rem; color: var(--text-dim); width: 100%;">Sample Claims:</span>
                            <button type="button" class="btn-sample" onclick="loadSample('Metformin is used as first-line therapy for type 2 diabetes mellitus.', 'healthcare')">Diabetes Drug</button>
                            <button type="button" class="btn-sample" onclick="loadSample('Log4Shell is associated with vulnerability CVE-2021-44228.', 'cybersecurity')">CVE Vulnerability</button>
                            <button type="button" class="btn-sample" onclick="loadSample('Apple Inc. trades under ticker AAPL on NASDAQ.', 'finance')">SEC Filing</button>
                            <button type="button" class="btn-sample" onclick="loadSample('Vitamin C completely cures all viral infections including influenza.', 'healthcare')">False Medical Claim</button>
                        </div>
                    </div>

                    <button type="submit" class="btn-primary" id="btnVerify">
                        <span>Verify Claim</span>
                        <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                    </button>
                </form>
            </div>
        </section>

        <!-- Right Column: Live Results -->
        <section class="results-container" id="resultsContainer">
            <div class="card empty-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                <h3>Ready to Verify</h3>
                <p>Select a sample claim or enter your own text, then click <strong>Verify Claim</strong> to execute the 9-stage pipeline live.</p>
            </div>
        </section>
    </main>

    <script>
        function loadSample(text, domain) {
            document.getElementById('claimText').value = text;
            document.getElementById('domainSelect').value = domain;
        }

        document.getElementById('verifyForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const claimText = document.getElementById('claimText').value.trim();
            const domain = document.getElementById('domainSelect').value;
            const btn = document.getElementById('btnVerify');
            const container = document.getElementById('resultsContainer');

            if (!claimText) return;

            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Running 9-Stage Pipeline...';

            try {
                const response = await fetch('/verify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query_id: 'ui-' + Date.now(),
                        domain: domain,
                        suspicious_claims: [{ claim_id: 'c1', text: claimText }]
                    })
                });

                const data = await response.json();
                renderResults(data);
            } catch (err) {
                container.innerHTML = `<div class="card" style="color: var(--hallucinated-red);">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>Verify Claim</span><svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>';
            }
        });

        function renderResults(data) {
            const report = data.claim_evidence[0] || {};
            const verdict = report.verdict || 'insufficient_evidence';
            const evidenceItems = report.evidence || [];
            const runtime = data.runtime_models || {};

            let html = `
                <!-- Verdict Banner -->
                <div class="verdict-banner ${verdict}">
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Verification Verdict</div>
                        <div class="verdict-tag">${verdict.replace('_', ' ')}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Overall Confidence</div>
                        <div style="font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${(data.overall_evidence_confidence * 100).toFixed(1)}%</div>
                    </div>
                </div>

                <!-- Metrics Grid -->
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-label">Trust Score</div>
                        <div class="metric-value" style="color: var(--accent-cyan);">${(report.trust_score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Support Score</div>
                        <div class="metric-value" style="color: var(--verified-green);">${(report.support_score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Contradiction</div>
                        <div class="metric-value" style="color: var(--hallucinated-red);">${(report.contradiction_score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">${data.latency_ms}ms</div>
                    </div>
                </div>

                <!-- Runtime Models Metadata -->
                <div class="runtime-info">
                    <div class="runtime-item"><span>Embedding Model</span><strong>${runtime.embedding_model || 'bge-m3'}</strong></div>
                    <div class="runtime-item"><span>Reranker</span><strong>${runtime.reranker_model || 'ms-marco-MiniLM'}</strong></div>
                    <div class="runtime-item"><span>NLI Model</span><strong>${runtime.nli_model || 'nli-deberta-v3'}</strong></div>
                </div>

                <!-- Natural Language Explanation -->
                <div class="card">
                    <div class="card-header">
                        <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        Natural Language Explanation
                    </div>
                    <div class="explanation-box">${report.explanation || 'No explanation generated.'}</div>
                </div>

                <!-- Evidence Citations -->
                <div class="card">
                    <div class="card-header">
                        <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6m-6 4h6"/></svg>
                        Authoritative Evidence Passages (${evidenceItems.length})
                    </div>
            `;

            if (evidenceItems.length === 0) {
                html += `<div style="color: var(--text-dim); text-align: center; padding: 2rem;">No evidence passages retrieved.</div>`;
            } else {
                evidenceItems.forEach(item => {
                    const badgeClass = item.entailment_label === 'entailment' ? 'badge-entailment' : (item.entailment_label === 'contradiction' ? 'badge-contradiction' : 'badge-neutral');
                    html += `
                        <div class="evidence-card">
                            <div class="evidence-header">
                                <a href="${item.url}" target="_blank" class="evidence-title">${item.title}</a>
                                <span class="entailment-badge ${badgeClass}">${item.entailment_label}</span>
                            </div>
                            <div class="evidence-snippet">"${item.snippet}"</div>
                            <div class="evidence-meta">
                                <span>Source: <strong>${item.source}</strong></span>
                                <span>Credibility: <strong>${(item.credibility_score * 100).toFixed(0)}%</strong></span>
                                <span>Entailment Score: <strong>${(item.entailment_score * 100).toFixed(1)}%</strong></span>
                                <span>Date: <strong>${item.publication_date}</strong></span>
                            </div>
                        </div>
                    `;
                });
            }

            html += `</div>`;
            document.getElementById('resultsContainer').innerHTML = html;
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response

def _get_pipeline(request: Request) -> VerificationPipeline:
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        pipeline = VerificationPipeline()
        request.app.state.pipeline = pipeline
    return request.app.state.pipeline

@app.post("/verify", response_model=VerifierOutputV2)
async def verify(payload: VerifierInputV2, request: Request) -> VerifierOutputV2:
    pipeline = _get_pipeline(request)
    try:
        response = await pipeline.verify(payload)
        rec = VerificationLogRecord(
            query_id=payload.query_id,
            request_id=payload.query_id,
            domain=payload.domain,
            stages_timing={},
            adapters_called=[payload.domain],
            adapters_failed=[],
            verdict=response.claim_evidence[0].verdict.value if response.claim_evidence else "unverified",
            confidence=response.overall_evidence_confidence,
            total_latency_ms=response.latency_ms,
            timestamp=time.time()
        )
        log_verification_request(logger, rec)
        return response
    except Exception as e:
        logger.error(f"Error during verification endpoint call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    registry = get_registry()
    model_manager = get_model_manager()
    
    try:
        cache_stats = await pipeline.cache.stats()
    except Exception:
        cache_stats = {"status": "unavailable"}

    domain_registry = get_domain_intelligence_registry()

    return {
        "status": "healthy",
        "version": get_version_info(),
        "models": model_manager.status(),
        "domain_intelligence": {
            "version": domain_registry.version,
            "configured_domains": len(domain_registry.list_domains()),
            "configured_model_ids": len(domain_registry.unique_model_ids()),
        },
        "cache": cache_stats,
        "adapters_registered": registry.list_domains(),
        "uptime_seconds": round(time.time() - _start_time, 2)
    }

@app.get("/domains", response_model=List[DomainStatistics])
async def domains() -> List[DomainStatistics]:
    registry = get_registry()
    all_stats = registry.get_all_statistics()
    result: List[DomainStatistics] = []
    for stat in all_stats:
        result.append(DomainStatistics(
            domain=stat.get("domain", "unknown"),
            sources=stat.get("sources", []),
            status=stat.get("status", "active"),
            credibility_scores=stat.get("credibility_scores", {}),
            is_implemented=stat.get("is_implemented", True),
            adapter_health=[]
        ))
    return result

@app.get("/pipeline")
async def get_pipeline_info() -> Dict[str, Any]:
    return {
        "stages": [
            {"name": "Domain Validation", "order": 1, "description": "Validates domain classification using zero-shot inference."},
            {"name": "Claim Decomposition", "order": 2, "description": "Decomposes compound claims into atomic sub-claims."},
            {"name": "Query Expansion", "order": 3, "description": "Expands sub-claims with domain-specific terminology."},
            {"name": "Multi-source Retrieval", "order": 4, "description": "Fetches evidence from domain-specific APIs in parallel."},
            {"name": "Aggregation + Dedup", "order": 5, "description": "Combines sources & removes >85% Jaccard token overlap duplicates."},
            {"name": "Hybrid RRF Retrieval", "order": 6, "description": "Reciprocal Rank Fusion of BM25 + FAISS Dense vector search."},
            {"name": "Cross-encoder Reranking", "order": 7, "description": "Scores passages against claims using ms-marco-MiniLM."},
            {"name": "DeBERTa NLI Entailment", "order": 8, "description": "Predicts logical entailment (Entailment/Contradiction/Neutral)."},
            {"name": "Evidence Scoring", "order": 9, "description": "Applies credibility weighting and recency decay factors."},
            {"name": "Conflict Resolution", "order": 10, "description": "Resolves contradictory evidence with 2:1 majority logic."},
            {"name": "Explanation Generation", "order": 11, "description": "Produces human-readable natural language explanations."},
            {"name": "Citation Formatting", "order": 12, "description": "Formats metadata citations adhering to VerifierOutputV2 contract."}
        ],
        "last_execution": {}
    }

@app.get("/metrics")
async def metrics(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    return pipeline.metrics.get_summary()
