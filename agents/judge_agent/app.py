"""
HalluciGuard - Judge Agent Interactive Web Dashboard
Provides an interactive GUI to test user queries, ChatGPT draft responses,
and simulated Verifier evidence against the Judge Agent.
"""

import json
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

from judge_agent import JudgeAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HalluciGuard.WebApp")

PORT = 8080
judge_agent_instance = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HalluciGuard | Judge Agent Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #131b2e;
            --panel-border: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.25);
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --accept: #10b981;
            --correct: #f59e0b;
            --reject: #ef4444;
            --abstain: #64748b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            min-height: 100vh;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 20px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--panel-border);
        }

        .badge-tag {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #cbd5e1; display: flex; align-items: center; gap: 8px; }

        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 12px; font-weight: 500; color: var(--text-dim); margin-bottom: 6px; }
        input, textarea, select {
            width: 100%;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 10px 12px;
            color: #fff;
            font-family: inherit;
            font-size: 13px;
        }
        input:focus, textarea:focus { outline: none; border-color: var(--accent); }

        .btn {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 14px var(--accent-glow);
        }
        .btn:hover { transform: translateY(-1px); opacity: 0.95; }

        .decision-badge {
            display: inline-block;
            font-size: 18px;
            font-weight: 800;
            padding: 8px 16px;
            border-radius: 8px;
            letter-spacing: 0.5px;
            margin-bottom: 14px;
        }
        .ACCEPT { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
        .CORRECT { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
        .REJECT { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
        .VERIFY_AGAIN { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid #6366f1; }
        .ABSTAIN { background: rgba(100, 116, 139, 0.2); color: #cbd5e1; border: 1px solid #64748b; }

        .metric-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }
        .metric-bar-bg { background: rgba(255,255,255,0.08); height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 12px; }
        .metric-bar-fill { height: 100%; background: var(--accent); transition: width 0.4s ease; }

        pre {
            background: #090d16;
            padding: 12px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #a7f3d0;
            overflow-x: auto;
            max-height: 250px;
        }

        .preset-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--panel-border);
            color: #94a3b8;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
            margin-right: 6px;
            margin-bottom: 8px;
        }
        .preset-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>HalluciGuard <span style="font-weight: 300; color: #94a3b8;">| Judge Agent</span></h1>
            <p style="font-size: 13px; color: #64748b; margin-top: 4px;">Multi-Agent Hallucination Mitigation & Confidence Calibration</p>
        </div>
        <span class="badge-tag">Role: Judge Agent</span>
    </header>

    <div style="margin-bottom: 16px;">
        <span style="font-size: 12px; color: var(--text-dim);">Load Preset Test Cases:</span><br>
        <button class="preset-btn" onclick="loadPreset('accurate')">Factually Accurate (Accept)</button>
        <button class="preset-btn" onclick="loadPreset('hallucination')">Medical Hallucination (Correct)</button>
        <button class="preset-btn" onclick="loadPreset('low_evidence')">Unverifiable / Missing Evidence (Abstain)</button>
    </div>

    <div class="grid">
        <!-- Input Panel -->
        <div class="card">
            <h2>Inputs from Verifier & ChatGPT / User</h2>
            
            <div class="form-group">
                <label>User Query / Context</label>
                <input type="text" id="user_query" value="What is the recommended pediatric fever treatment?">
            </div>

            <div class="form-group">
                <label>ChatGPT Draft Response (Claim under Evaluation)</label>
                <textarea id="draft_response" rows="3">Aspirin is the recommended first-line treatment for children with viral fever or flu.</textarea>
            </div>

            <div class="form-group">
                <label>Verifier Ground-Truth Evidence</label>
                <textarea id="verifier_evidence" rows="3">Aspirin is strictly contraindicated in children recovering from viral fever due to severe risk of Reye's syndrome. Acetaminophen or Ibuprofen should be used.</textarea>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div class="form-group">
                    <label>Domain</label>
                    <select id="domain">
                        <option value="Healthcare">Healthcare</option>
                        <option value="Finance">Finance</option>
                        <option value="Law">Law</option>
                        <option value="Cybersecurity">Cybersecurity</option>
                        <option value="General">General</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Detector Risk Signal (Prob: 0 to 1)</label>
                    <input type="number" id="detector_prob" step="0.05" min="0" max="1" value="0.80">
                </div>
            </div>

            <button class="btn" onclick="evaluateJudge()">Run Judge Agent Evaluation</button>
        </div>

        <!-- Output Panel -->
        <div class="card">
            <h2>Judge Agent Evaluation & Corrector Payload</h2>
            
            <div id="output-placeholder" style="color: #64748b; text-align: center; padding: 40px 0;">
                Click "Run Judge Agent Evaluation" to process signals.
            </div>

            <div id="output-results" style="display: none;">
                <div id="decision-badge" class="decision-badge ACCEPT">ACCEPT</div>

                <div class="metric-row">
                    <span>Calibrated Confidence</span>
                    <strong id="conf-val">0.00</strong>
                </div>
                <div class="metric-bar-bg"><div id="conf-bar" class="metric-bar-fill" style="width: 0%;"></div></div>

                <div class="metric-row">
                    <span>Contradiction Index</span>
                    <strong id="contra-val">0.00</strong>
                </div>
                <div class="metric-bar-bg"><div id="contra-bar" class="metric-bar-fill" style="width: 0%; background: #ef4444;"></div></div>

                <p style="font-size: 13px; margin-bottom: 8px;"><strong>Reason:</strong> <span id="reason-text" style="color: #cbd5e1;"></span></p>
                <p style="font-size: 13px; margin-bottom: 14px;"><strong>Next Action:</strong> <span id="action-text" style="color: #a5b4fc;"></span></p>

                <h3 style="font-size: 13px; margin-bottom: 6px; color: #94a3b8;">Payload Emitted to Corrector Agent</h3>
                <pre id="corrector-json"></pre>
            </div>
        </div>
    </div>

    <script>
        function loadPreset(type) {
            if (type === 'accurate') {
                document.getElementById('user_query').value = "What was Apple's total revenue in FY 2023?";
                document.getElementById('draft_response').value = "Apple reported net sales revenue of $383.29 billion in fiscal year 2023.";
                document.getElementById('verifier_evidence').value = "Apple Inc. SEC 10-K filing reports total annual revenue of $383.29 billion for FY 2023.";
                document.getElementById('domain').value = "Finance";
                document.getElementById('detector_prob').value = "0.05";
            } else if (type === 'hallucination') {
                document.getElementById('user_query').value = "What is the recommended pediatric fever treatment?";
                document.getElementById('draft_response').value = "Aspirin is the recommended first-line treatment for children with viral fever or flu.";
                document.getElementById('verifier_evidence').value = "Aspirin is strictly contraindicated in children recovering from viral fever due to severe risk of Reye's syndrome. Acetaminophen or Ibuprofen should be used.";
                document.getElementById('domain').value = "Healthcare";
                document.getElementById('detector_prob').value = "0.85";
            } else if (type === 'low_evidence') {
                document.getElementById('user_query').value = "Does CVE-2026-0001 affect Windows 11?";
                document.getElementById('draft_response').value = "CVE-2026-0001 allows kernel privilege escalation on default installations.";
                document.getElementById('verifier_evidence').value = "";
                document.getElementById('domain').value = "Cybersecurity";
                document.getElementById('detector_prob').value = "0.60";
            }
            evaluateJudge();
        }

        async function evaluateJudge() {
            const payload = {
                user_query: document.getElementById('user_query').value,
                draft_response: document.getElementById('draft_response').value,
                verifier_evidence: document.getElementById('verifier_evidence').value,
                domain: document.getElementById('domain').value,
                detector_prob: parseFloat(document.getElementById('detector_prob').value)
            };

            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const res = await response.json();
            
            document.getElementById('output-placeholder').style.display = 'none';
            document.getElementById('output-results').style.display = 'block';

            const badge = document.getElementById('decision-badge');
            badge.className = 'decision-badge ' + res.decision;
            badge.innerText = res.decision + ' (Severity: ' + res.severity + ')';

            const conf = res.metrics.calibrated_confidence;
            document.getElementById('conf-val').innerText = (conf * 100).toFixed(1) + '%';
            document.getElementById('conf-bar').style.width = (conf * 100) + '%';

            const contra = res.metrics.overall_contradiction;
            document.getElementById('contra-val').innerText = (contra * 100).toFixed(1) + '%';
            document.getElementById('contra-bar').style.width = (contra * 100) + '%';

            document.getElementById('reason-text').innerText = res.reason;
            document.getElementById('action-text').innerText = res.next_action;

            document.getElementById('corrector-json').innerText = JSON.stringify(res.corrector_payload, null, 2);
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            user_query = data.get("user_query", "")
            draft_response = data.get("draft_response", "")
            verifier_evidence = data.get("verifier_evidence", "")
            domain = data.get("domain", "General")
            detector_prob = float(data.get("detector_prob", 0.3))

            detector_output = {
                "hallucination_probability": detector_prob,
                "confidence_score": 0.85,
                "risk_level": "HIGH" if detector_prob > 0.5 else "LOW"
            }

            verifier_output = {
                "domain": domain,
                "claim_evidence_pairs": [
                    {
                        "claim": draft_response,
                        "evidence": verifier_evidence,
                        "evidence_confidence": 0.90 if verifier_evidence else 0.0,
                        "rank": 1,
                        "source": f"Verifier KB ({domain})"
                    }
                ] if verifier_evidence else []
            }

            result = judge_agent_instance.evaluate(
                detector_output=detector_output,
                verifier_output=verifier_output,
                user_query=user_query,
                draft_response=draft_response
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_web_server():
    global judge_agent_instance
    print("Initializing Judge Agent...")
    judge_agent_instance = JudgeAgent()
    
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"\n=======================================================")
    print(f" HalluciGuard Judge Agent Web Dashboard Running")
    print(f" Open your browser to: http://localhost:{PORT}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")

if __name__ == "__main__":
    run_web_server()
