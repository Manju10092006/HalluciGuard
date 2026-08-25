"""
HalluciGuard Judge Agent - Enterprise Decision Intelligence Dashboard
Real-time interactive web dashboard for the AI Operating System.
"""

import sys
import os
import json

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify
from decision_intelligence import DecisionIntelligenceEngine

app = Flask(__name__)
engine = DecisionIntelligenceEngine()

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HalluciGuard | Judge Agent — AI Decision Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0b0f;--surface:#12131a;--surface2:#1a1c27;--surface3:#22253a;
  --border:#2a2d42;--border-glow:#3b3f60;
  --text:#e4e6f0;--text2:#9da3c0;--text3:#6b7194;
  --accent:#6366f1;--accent2:#818cf8;--accent-glow:rgba(99,102,241,.15);
  --green:#10b981;--green-bg:rgba(16,185,129,.08);--green-border:rgba(16,185,129,.25);
  --red:#ef4444;--red-bg:rgba(239,68,68,.08);--red-border:rgba(239,68,68,.25);
  --yellow:#f59e0b;--yellow-bg:rgba(245,158,11,.08);--yellow-border:rgba(245,158,11,.25);
  --blue:#3b82f6;--blue-bg:rgba(59,130,246,.08);--blue-border:rgba(59,130,246,.25);
  --cyan:#06b6d4;--purple:#a855f7;
  --radius:12px;--radius-sm:8px;--radius-xs:6px;
}
html{font-size:14px}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5}

/* ── Header ── */
.header{background:linear-gradient(135deg,#12131a 0%,#1a1040 50%,#12131a 100%);border-bottom:1px solid var(--border);padding:20px 32px;display:flex;align-items:center;gap:16px}
.header .logo{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--purple));display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;color:#fff}
.header h1{font-size:1.4rem;font-weight:700;background:linear-gradient(135deg,#e4e6f0,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .subtitle{font-size:.78rem;color:var(--text3);font-weight:400;margin-left:auto}

/* ── Layout ── */
.container{display:grid;grid-template-columns:380px 1fr;gap:0;min-height:calc(100vh - 80px)}
.panel{padding:24px;overflow-y:auto;max-height:calc(100vh - 80px)}
.left-panel{background:var(--surface);border-right:1px solid var(--border)}
.right-panel{background:var(--bg);padding:24px 28px}

/* ── Section ── */
.section{margin-bottom:24px}
.section-title{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--text3);margin-bottom:12px;display:flex;align-items:center;gap:6px}
.section-title::before{content:'';width:3px;height:14px;border-radius:2px;background:var(--accent)}

/* ── Form Controls ── */
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:.75rem;font-weight:600;color:var(--text2);margin-bottom:5px}
textarea,select,input[type=text],input[type=number]{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-sm);color:var(--text);padding:10px 12px;font-family:'Inter',sans-serif;font-size:.82rem;transition:border-color .2s,box-shadow .2s;resize:vertical}
textarea:focus,select:focus,input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow)}
textarea{min-height:70px}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7194' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;cursor:pointer}

/* ── Slider ── */
.slider-group{margin-bottom:14px}
.slider-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.slider-header label{font-size:.75rem;font-weight:600;color:var(--text2)}
.slider-value{background:var(--surface3);border:1px solid var(--border);border-radius:var(--radius-xs);padding:2px 10px;font-size:.78rem;font-weight:700;color:var(--accent2);min-width:50px;text-align:center}
input[type=range]{-webkit-appearance:none;width:100%;height:6px;border-radius:3px;background:var(--surface3);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--accent);border:2px solid #fff;box-shadow:0 0 8px rgba(99,102,241,.4);cursor:pointer;transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15)}

/* ── Evidence Builder ── */
.evidence-pair{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;margin-bottom:8px;position:relative}
.evidence-pair input,.evidence-pair textarea{margin-bottom:6px;min-height:40px}
.pair-remove{position:absolute;top:6px;right:8px;background:none;border:none;color:var(--text3);cursor:pointer;font-size:.9rem}
.pair-remove:hover{color:var(--red)}
.add-evidence{width:100%;background:var(--surface3);border:1px dashed var(--border);border-radius:var(--radius-sm);color:var(--text3);padding:8px;cursor:pointer;font-size:.78rem;transition:all .2s}
.add-evidence:hover{border-color:var(--accent);color:var(--accent)}

/* ── Button ── */
.btn-primary{width:100%;padding:14px 24px;background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;font-weight:700;font-size:.9rem;border:none;border-radius:var(--radius);cursor:pointer;transition:all .25s;letter-spacing:.03em;position:relative;overflow:hidden}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 8px 30px rgba(99,102,241,.3)}
.btn-primary:active{transform:translateY(0)}
.btn-primary.loading{opacity:.7;pointer-events:none}
.btn-primary.loading::after{content:'';position:absolute;top:50%;left:50%;width:20px;height:20px;margin:-10px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Decision Badge ── */
.decision-display{text-align:center;padding:28px 20px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:20px;position:relative;overflow:hidden}
.decision-display::before{content:'';position:absolute;inset:0;background:radial-gradient(circle at 50% 0%,var(--accent-glow),transparent 70%);pointer-events:none}
.decision-badge{display:inline-flex;align-items:center;gap:10px;padding:12px 28px;border-radius:40px;font-weight:800;font-size:1.1rem;letter-spacing:.06em}
.decision-ACCEPT{background:var(--green-bg);border:2px solid var(--green-border);color:var(--green)}
.decision-CORRECT{background:var(--yellow-bg);border:2px solid var(--yellow-border);color:var(--yellow)}
.decision-REJECT{background:var(--red-bg);border:2px solid var(--red-border);color:var(--red)}
.decision-VERIFY_AGAIN{background:var(--blue-bg);border:2px solid var(--blue-border);color:var(--blue)}
.decision-ABSTAIN{background:rgba(107,113,148,.1);border:2px solid rgba(107,113,148,.25);color:var(--text3)}
.decision-ESCALATE_HUMAN{background:rgba(168,85,247,.1);border:2px solid rgba(168,85,247,.25);color:var(--purple)}
.decision-icon{font-size:1.4rem}
.severity-tag{margin-top:10px;display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.1em;padding:4px 14px;border-radius:20px;background:var(--surface3);color:var(--text2)}

/* ── Metric Cards ── */
.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:14px;text-align:center}
.metric-label{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);margin-bottom:4px}
.metric-value{font-size:1.1rem;font-weight:800}
.metric-value.good{color:var(--green)}.metric-value.warn{color:var(--yellow)}.metric-value.bad{color:var(--red)}.metric-value.info{color:var(--blue)}

/* ── Reasoning Panel ── */
.reasoning-chain{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:16px}
.reasoning-step{padding:8px 12px;border-left:3px solid var(--border);margin-bottom:6px;font-size:.78rem;color:var(--text2);line-height:1.6;transition:border-color .2s}
.reasoning-step:hover{border-color:var(--accent);color:var(--text)}
.reasoning-step strong{color:var(--text)}

/* ── Claim Table ── */
.claims-table{width:100%;border-collapse:separate;border-spacing:0;margin-bottom:16px}
.claims-table th{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
.claims-table td{padding:8px 10px;font-size:.78rem;border-bottom:1px solid rgba(42,45,66,.5);color:var(--text2);vertical-align:top}
.claims-table tr:hover td{background:rgba(99,102,241,.03)}
.status-pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.68rem;font-weight:700;white-space:nowrap}
.status-VERIFIED{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.status-HALLUCINATED{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border)}
.status-UNVERIFIED{background:var(--yellow-bg);color:var(--yellow);border:1px solid var(--yellow-border)}

/* ── Collapsible ── */
.collapsible{margin-bottom:12px}
.collapsible-header{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:.78rem;font-weight:600;color:var(--text);transition:all .2s}
.collapsible-header:hover{border-color:var(--accent)}
.collapsible-body{display:none;padding:12px 14px;background:var(--surface);border:1px solid var(--border);border-top:none;border-radius:0 0 var(--radius-sm) var(--radius-sm);font-size:.76rem;color:var(--text2);line-height:1.7}
.collapsible.open .collapsible-body{display:block}
.collapsible-chevron{transition:transform .2s;font-size:.7rem;color:var(--text3)}
.collapsible.open .collapsible-chevron{transform:rotate(180deg)}

/* ── Empty State ── */
.empty-state{text-align:center;padding:80px 40px;color:var(--text3)}
.empty-state .icon{font-size:3rem;margin-bottom:16px;opacity:.4}
.empty-state h3{font-size:1.1rem;color:var(--text2);margin-bottom:8px}
.empty-state p{font-size:.82rem}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:var(--border-glow)}

/* ── Responsive ── */
@media(max-width:900px){.container{grid-template-columns:1fr}.left-panel{border-right:none;border-bottom:1px solid var(--border)}.panel{max-height:none}}
</style>
</head>
<body>

<div class="header">
  <div class="logo">J</div>
  <div>
    <h1>HalluciGuard Judge Agent</h1>
    <div style="font-size:.72rem;color:var(--text3)">AI Decision Intelligence Platform — Chief Decision Officer</div>
  </div>
  <div class="subtitle">v3.0 | Enterprise Governance Engine</div>
</div>

<div class="container">
  <!-- LEFT: Input Panel -->
  <div class="panel left-panel">
    <div class="section">
      <div class="section-title">Simulation Input</div>

      <div class="form-group">
        <label>User Query</label>
        <textarea id="userQuery" placeholder="What question was asked?">What is the recommended dosage of ibuprofen for adults?</textarea>
      </div>

      <div class="form-group">
        <label>Draft LLM Response</label>
        <textarea id="draftResponse" placeholder="The LLM's response to evaluate..." style="min-height:90px">The recommended adult dosage of ibuprofen is 200-400mg every 4-6 hours. Do not exceed 3200mg per day. Take with food to reduce stomach upset.</textarea>
      </div>

      <div class="form-group">
        <label>Domain</label>
        <select id="domain">
          <option value="Healthcare">Healthcare</option>
          <option value="Cybersecurity">Cybersecurity</option>
          <option value="Finance">Finance</option>
          <option value="Law">Law</option>
          <option value="Scientific Research">Scientific Research</option>
          <option value="General Knowledge" selected>General Knowledge</option>
          <option value="Entertainment">Entertainment</option>
        </select>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Detector Agent Signal</div>
      <div class="slider-group">
        <div class="slider-header">
          <label>Hallucination Probability</label>
          <span class="slider-value" id="detProbValue">0.30</span>
        </div>
        <input type="range" id="detProb" min="0" max="1" step="0.01" value="0.30" oninput="document.getElementById('detProbValue').textContent=parseFloat(this.value).toFixed(2)">
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <label>Detector Confidence</label>
          <span class="slider-value" id="detConfValue">0.85</span>
        </div>
        <input type="range" id="detConf" min="0" max="1" step="0.01" value="0.85" oninput="document.getElementById('detConfValue').textContent=parseFloat(this.value).toFixed(2)">
      </div>
    </div>

    <div class="section">
      <div class="section-title">Verifier Agent Evidence</div>
      <div id="evidencePairs">
        <div class="evidence-pair" data-idx="0">
          <button class="pair-remove" onclick="removePair(this)">✕</button>
          <input type="text" class="ev-claim" placeholder="Claim..." value="The recommended adult dosage of ibuprofen is 200-400mg every 4-6 hours.">
          <textarea class="ev-evidence" placeholder="Evidence snippet...">Adults and children over 12: take 200mg to 400mg of ibuprofen every 4 to 6 hours as needed. Do not take more than 1200mg in 24 hours unless directed by a doctor.</textarea>
          <input type="text" class="ev-source" placeholder="Source..." value="NHS Official Guidelines">
        </div>
        <div class="evidence-pair" data-idx="1">
          <button class="pair-remove" onclick="removePair(this)">✕</button>
          <input type="text" class="ev-claim" placeholder="Claim..." value="Do not exceed 3200mg per day.">
          <textarea class="ev-evidence" placeholder="Evidence snippet...">The maximum daily dose of ibuprofen for adults is 1200mg for over-the-counter use. Prescription doses may go up to 3200mg per day under medical supervision.</textarea>
          <input type="text" class="ev-source" placeholder="Source..." value="FDA Drug Label">
        </div>
      </div>
      <button class="add-evidence" onclick="addPair()">+ Add Evidence Pair</button>
    </div>

    <div style="display:flex;gap:8px;margin-top:10px">
      <button class="btn-primary" id="evaluateBtn" onclick="runDecisionPipeline()">
        ⚡ Execute Decision Pipeline
      </button>
      <button style="width:130px;background:var(--surface3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text2);font-weight:600;font-size:.78rem;cursor:pointer" onclick="clearForm()">
        🗑 Clear Form
      </button>
    </div>
  </div>

  <!-- RIGHT: Results Panel -->
  <div class="panel right-panel" id="resultsPanel">
    <div class="empty-state" id="emptyState">
      <div class="icon">⚖️</div>
      <h3>Decision Intelligence Awaiting Input</h3>
      <p>Configure the simulation parameters and execute the pipeline to see the Judge Agent's governance reasoning.</p>
    </div>
    <div id="resultsContent" style="display:none"></div>
  </div>
</div>

<script>
let pairIdx = 2;
function addPair() {
  const div = document.createElement('div');
  div.className = 'evidence-pair';
  div.dataset.idx = pairIdx++;
  div.innerHTML = `<button class="pair-remove" onclick="removePair(this)">✕</button>
    <input type="text" class="ev-claim" placeholder="Claim...">
    <textarea class="ev-evidence" placeholder="Evidence snippet..."></textarea>
    <input type="text" class="ev-source" placeholder="Source...">`;
  document.getElementById('evidencePairs').appendChild(div);
}
function removePair(btn) {
  const pairs = document.querySelectorAll('.evidence-pair');
  if (pairs.length > 1) btn.closest('.evidence-pair').remove();
}

function collectPairs() {
  const pairs = [];
  document.querySelectorAll('.evidence-pair').forEach(p => {
    const cl = p.querySelector('.ev-claim').value.trim();
    const ev = p.querySelector('.ev-evidence').value.trim();
    const src = p.querySelector('.ev-source').value.trim();
    if (cl || ev) {
      pairs.push({ claim: cl, evidence: ev, source: src || "Unverified Source" });
    }
  });
  return pairs;
}

function clearForm() {
  document.getElementById('userQuery').value = '';
  document.getElementById('draftResponse').value = '';
  document.getElementById('detProb').value = '0.50';
  document.getElementById('detProbValue').textContent = '0.50';
  document.getElementById('detConf').value = '0.80';
  document.getElementById('detConfValue').textContent = '0.80';
  document.getElementById('evidencePairs').innerHTML = `
    <div class="evidence-pair" data-idx="0">
      <button class="pair-remove" onclick="removePair(this)">✕</button>
      <input type="text" class="ev-claim" placeholder="Claim (Optional - auto-extracted from response)...">
      <textarea class="ev-evidence" placeholder="Evidence snippet (Optional)..."></textarea>
      <input type="text" class="ev-source" placeholder="Source (Optional)...">
    </div>`;
  document.getElementById('resultsContent').style.display = 'none';
  document.getElementById('emptyState').style.display = 'block';
}

async function runDecisionPipeline() {
  const btn = document.getElementById('evaluateBtn');
  btn.classList.add('loading');
  btn.textContent = '';

  try {
    const resp = await fetch('/evaluate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        user_query: document.getElementById('userQuery').value,
        draft_response: document.getElementById('draftResponse').value,
        domain: document.getElementById('domain').value,
        detector_output: {
          hallucination_probability: parseFloat(document.getElementById('detProb').value),
          confidence_score: parseFloat(document.getElementById('detConf').value)
        },
        verifier_output: {
          claim_evidence_pairs: collectPairs()
        }
      })
    });
    const data = await resp.json();
    renderResults(data);
  } catch (e) {
    alert('Error executing pipeline: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = '⚡ Execute Decision Pipeline';
  }
}

// Auto-run pipeline on initial page load
window.addEventListener('DOMContentLoaded', runDecisionPipeline);

function renderResults(d) {
  document.getElementById('emptyState').style.display = 'none';
  const rc = document.getElementById('resultsContent');
  rc.style.display = 'block';

  const decIcon = {ACCEPT:'✅',CORRECT:'🔧',REJECT:'🛑',VERIFY_AGAIN:'🔄',ABSTAIN:'⏸️',ESCALATE_HUMAN:'👨‍⚕️'}[d.decision]||'❓';

  // Metric helpers
  function mClass(val, good, warn) {
    if (typeof val === 'boolean') return val ? 'good' : 'bad';
    if (typeof val === 'string') {
      if (['HEALTHY','FULLY_COVERED','STRONG_AGREEMENT','AUTHORITATIVE','STRONG'].includes(val)) return 'good';
      if (['DEGRADED','PARTIALLY_COVERED','MAJORITY_AGREEMENT','MODERATE'].includes(val)) return 'warn';
      return 'bad';
    }
    return 'info';
  }

  let claimRows = '';
  if (d.claim_decisions && d.claim_decisions.length) {
    d.claim_decisions.forEach(cd => {
      const actClass = cd.action === 'ACCEPT' ? 'status-VERIFIED' : (cd.action === 'CORRECT' ? 'status-HALLUCINATED' : 'status-UNVERIFIED');
      const stClass = cd.status === 'VERIFIED' ? 'status-VERIFIED' : (cd.status === 'CONTRADICTED' ? 'status-HALLUCINATED' : 'status-UNVERIFIED');
      const evIds = (cd.evidence_ids || []).join(', ') || '—';

      claimRows += `<tr>
        <td style="font-weight:700;color:var(--accent2)">${cd.claim_id || '—'}</td>
        <td><span class="status-pill ${stClass}">${cd.status}</span></td>
        <td><span class="status-pill ${actClass}" style="font-weight:800">${cd.action}</span></td>
        <td>${cd.claim_text ? cd.claim_text.substring(0,80) : '—'}${cd.claim_text && cd.claim_text.length > 80 ? '…' : ''}</td>
        <td><span style="font-family:monospace;background:var(--surface3);padding:2px 6px;border-radius:4px;color:var(--cyan)">${evIds}</span></td>
        <td style="font-size:.72rem;color:var(--text3)">${cd.reason || '—'}</td>
      </tr>`;
    });
  }

  let reasoningSteps = '';
  if (d.reasoning_chain) {
    d.reasoning_chain.forEach(s => {
      reasoningSteps += `<div class="reasoning-step">${s}</div>`;
    });
  }

  let riskFactors = '';
  if (d.risk_assessment && d.risk_assessment.factors) {
    riskFactors = d.risk_assessment.factors.map(f => `<div style="padding:4px 0;color:var(--red)">⚠ ${f}</div>`).join('');
    riskFactors += (d.risk_assessment.mitigating||[]).map(m => `<div style="padding:4px 0;color:var(--green)">✓ ${m}</div>`).join('');
  }

  let conflictsHtml = '';
  if (d.conflicts) {
    d.conflicts.forEach(c => {
      if (c.type === 'NO_CONFLICT') return;
      conflictsHtml += `<div style="padding:6px 0;border-bottom:1px solid var(--border)">
        <span style="color:${c.safety_critical?'var(--red)':'var(--yellow)'}">▪ [${c.type}]</span>
        ${c.implication}<br><small style="color:var(--text3)">${c.claim}</small></div>`;
    });
  }

  let workflowHtml = '';
  if (d.workflow_action) {
    const wa = d.workflow_action;
    workflowHtml = `<div style="padding:8px"><strong>Action:</strong> ${wa.type}<br><strong>Target:</strong> ${wa.target}<br><strong>Priority:</strong> ${wa.priority}<br><strong>Reasoning:</strong> ${wa.reasoning}</div>`;
  }

  let auditHtml = '';
  if (d.audit_record) {
    auditHtml = `<div style="font-size:.75rem;color:var(--text3)">Audit ID: ${d.audit_record.id}<br>Timestamp: ${d.audit_record.timestamp}</div>`;
  }

  rc.innerHTML = `
    <!-- Decision Badge -->
    <div class="decision-display">
      <div class="decision-badge decision-${d.decision}">
        <span class="decision-icon">${decIcon}</span> ${d.decision}
      </div>
      <div class="severity-tag">${d.severity}</div>
    </div>

    <!-- Metrics -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Evidence Quality</div>
        <div class="metric-value ${mClass(d.evidence_governance.quality)}">${d.evidence_governance.quality}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Pipeline Health</div>
        <div class="metric-value ${mClass(d.runtime_health.health)}">${d.runtime_health.health}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value ${mClass(d.risk_assessment.level==='INFORMATIONAL'||d.risk_assessment.level==='LOW'?'HEALTHY':d.risk_assessment.level==='MEDIUM'?'DEGRADED':'BAD')}">${d.risk_assessment.level}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Coverage</div>
        <div class="metric-value ${mClass(d.coverage.status)}">${d.coverage.status.replace(/_/g,' ')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Consensus</div>
        <div class="metric-value ${mClass(d.consensus.status)}">${d.consensus.status.replace(/_/g,' ')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Safe to Release</div>
        <div class="metric-value ${d.risk_assessment.safe_to_release?'good':'bad'}">${d.risk_assessment.safe_to_release?'YES':'NO'}</div>
      </div>
    </div>

    <!-- Detector Signal -->
    <div class="section">
      <div class="section-title">Detector Agent Signal (Input)</div>
      <div class="metrics-grid" style="grid-template-columns:1fr 1fr">
        <div class="metric-card">
          <div class="metric-label">Hallucination Probability</div>
          <div class="metric-value" style="color:${d.detector_signal.hallucination_probability>=0.5?'var(--red)':d.detector_signal.hallucination_probability>=0.3?'var(--yellow)':'var(--green)'}">${(d.detector_signal.hallucination_probability*100).toFixed(0)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Detector Confidence</div>
          <div class="metric-value info">${(d.detector_signal.confidence_score*100).toFixed(0)}%</div>
        </div>
      </div>
    </div>

    <!-- Claim Decisions -->
    ${claimRows ? `<div class="section">
      <div class="section-title">Claim-Level Governance & Action Decisions</div>
      <div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius-sm)">
        <table class="claims-table">
          <thead><tr><th>ID</th><th>Verifier Status</th><th>Judge Action</th><th>Claim Text</th><th>Evidence IDs</th><th>Governance Reason</th></tr></thead>
          <tbody>${claimRows}</tbody>
        </table>
      </div>
    </div>` : ''}

    <!-- Reasoning Chain -->
    <div class="section">
      <div class="section-title">Reasoning Chain</div>
      <div class="reasoning-chain">${reasoningSteps}</div>
    </div>

    <!-- Risk -->
    <div class="collapsible open">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Risk Assessment <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">${riskFactors || '<div style="color:var(--text3)">No significant risk factors.</div>'}<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:.76rem;color:var(--text2)">${d.risk_assessment.reasoning}</div></div>
    </div>

    <!-- Conflicts -->
    ${conflictsHtml ? `<div class="collapsible">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Conflict Analysis <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">${conflictsHtml}</div>
    </div>` : ''}

    <!-- Workflow Action -->
    <div class="collapsible">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Workflow Orchestration <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">${workflowHtml}</div>
    </div>

    <!-- Evidence Governance -->
    <div class="collapsible">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Evidence Governance Report <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">
        <div><strong>Quality:</strong> ${d.evidence_governance.quality} | <strong>Authoritative:</strong> ${d.evidence_governance.authoritative} | <strong>Sufficient:</strong> ${d.evidence_governance.sufficient?'Yes':'No'}</div>
        ${d.evidence_governance.concerns.length?'<div style="margin-top:8px"><strong>Concerns:</strong></div>'+d.evidence_governance.concerns.map(c=>'<div style="color:var(--yellow);padding:2px 0">⚠ '+c+'</div>').join(''):''}
        <div style="margin-top:8px;color:var(--text3)">${d.evidence_governance.reasoning}</div>
      </div>
    </div>

    <!-- Audit -->
    <div class="collapsible">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Audit Trail <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">${auditHtml}
        <pre style="margin-top:8px;padding:10px;background:var(--surface2);border-radius:var(--radius-xs);font-size:.72rem;color:var(--text2);overflow-x:auto;white-space:pre-wrap">${JSON.stringify(d.audit_record, null, 2)}</pre>
      </div>
    </div>

    <!-- Alternatives -->
    <div class="collapsible">
      <div class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">
        Alternatives Rejected <span class="collapsible-chevron">▼</span>
      </div>
      <div class="collapsible-body">
        ${Object.entries(d.alternatives_rejected||{}).map(([k,v])=>'<div style="padding:4px 0"><strong>'+k+':</strong> '+v+'</div>').join('')}
      </div>
    </div>
  `;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    res = engine.evaluate(
        user_query=data.get("user_query", ""),
        draft_response=data.get("draft_response", ""),
        detector_output=data.get("detector_output", {}),
        verifier_output=data.get("verifier_output", {}),
        domain=data.get("domain", ""),
        retry_count=data.get("retry_count", 0)
    )
    
    if hasattr(res, "model_dump"):
        dump = res.model_dump()
        decision_val = dump.get("decision", "ABSTAIN")
        severity_val = dump.get("severity", "LOW")
        corr_req = dump.get("correction_request")
        
        return jsonify({
            "decision": decision_val,
            "overall_decision": decision_val,
            "severity": severity_val,
            "reasoning_chain": [res.reason, res.explanation],
            "workflow_action": {
                "type": "REQUEST_CORRECTION" if decision_val == "CORRECT" else "ACCEPT_RELEASE",
                "target": "CORRECTOR" if decision_val == "CORRECT" else "USER",
                "instructions": corr_req.get("correction_instructions", "") if corr_req else ""
            },
            "risk_assessment": {"level": severity_val, "safe_to_release": decision_val == "ACCEPT"},
            "evidence_governance": {"quality": "AUTHORITATIVE", "sufficient": True, "reasoning": res.explanation, "concerns": []},
            "coverage": {},
            "conflicts": [],
            "consensus": {},
            "runtime_health": {"status": res.status},
            "memory_insight": {},
            "audit_record": {"decision": decision_val, "reason": res.reason},
            "claim_verdicts": [],
            "claim_decisions": corr_req.get("claims_to_correct", []) if corr_req else [],
            "correction_required": decision_val == "CORRECT",
            "re_verification_required": decision_val == "VERIFY_AGAIN",
            "detector_signal": data.get("detector_output", {}),
            "alternatives_rejected": {}
        })
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
