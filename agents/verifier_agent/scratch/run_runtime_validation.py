from __future__ import annotations
import asyncio
import json
import time
import subprocess
import os
import sys

# Ensure current directory is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress progress bars and verbose warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

from httpx import AsyncClient, ASGITransport
from api.main import app, lifespan
from models import get_model_manager
from adapters.registry import get_registry

async def main():
    print("=" * 80)
    print("1. RUNTIME VALIDATION & SERVER STARTUP LOGS")
    print("=" * 80)
    
    # Initialize lifespan
    print("[STARTUP] Triggering FastAPI app lifespan startup...")
    async with lifespan(app):
        print("[STARTUP] FastAPI Lifespan context entered successfully. Pipeline & SQLite Cache initialized.")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8002") as client:
            
            print("\n" + "=" * 80)
            print("2. ENDPOINT VALIDATION (HTTP API GET & POST)")
            print("=" * 80)
            
            # GET /health
            print("\n--- GET /health ---")
            res = await client.get("/health")
            print(f"Status Code: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
            
            # GET /domains
            print("\n--- GET /domains ---")
            res = await client.get("/domains")
            print(f"Status Code: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
            
            # GET /pipeline
            print("\n--- GET /pipeline ---")
            res = await client.get("/pipeline")
            print(f"Status Code: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
            
            # GET /metrics
            print("\n--- GET /metrics ---")
            res = await client.get("/metrics")
            print(f"Status Code: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
            
            # POST /verify Sample
            print("\n--- POST /verify (Sample Query) ---")
            sample_req = {
                "query_id": "req_verify_sample_001",
                "domain": "healthcare",
                "suspicious_claims": [
                    {
                        "claim_id": "claim_sample_1",
                        "text": "Metformin is a first-line oral medication for type 2 diabetes."
                    }
                ]
            }
            res = await client.post("/verify", json=sample_req)
            print(f"Status Code: {res.status_code}")
            print(json.dumps(res.json(), indent=2))
            
            print("\n" + "=" * 80)
            print("3. REAL VERIFICATION EXAMPLES (10 TEST CASES ACROSS 5 DOMAINS)")
            print("=" * 80)
            
            test_cases = [
                # Healthcare
                ("healthcare", "claim_hc_1", "Metformin is widely used as a first-line treatment for type 2 diabetes mellitus."),
                ("healthcare", "claim_hc_2", "Vitamin C consumption completely eliminates the need for insulin in type 1 diabetes."),
                
                # Cybersecurity
                ("cybersecurity", "claim_cs_1", "CVE-2021-44228 is a remote code execution vulnerability in Apache Log4j."),
                ("cybersecurity", "claim_cs_2", "SQL injection attacks are impossible when using a dynamic relational database."),
                
                # Finance
                ("finance", "claim_fn_1", "Publicly traded companies in the US file annual financial statements on SEC Form 10-K."),
                ("finance", "claim_fn_2", "Gross Domestic Product (GDP) measures the total monetary value of finished goods produced within a country."),
                
                # AI Research
                ("ai_research", "claim_ai_1", "Transformer neural networks use multi-head self-attention to encode sequence representations."),
                ("ai_research", "claim_ai_2", "Large Language Models are mathematically incapable of generating hallucinated statements."),
                
                # General
                ("general", "claim_gn_1", "Apollo 11 was the American spaceflight that first landed humans on the Moon in July 1969."),
                ("general", "claim_gn_2", "Domestic cats are cold-blooded reptiles that hibernate underwater during winter.")
            ]
            
            for idx, (domain, cid, claim_text) in enumerate(test_cases, 1):
                print(f"\n--------------------------------------------------------------------------------")
                print(f"EXAMPLE {idx:02d}/10 [{domain.upper()}]")
                print(f"Claim ID: {cid}")
                print(f"Input Claim: \"{claim_text}\"")
                print(f"--------------------------------------------------------------------------------")
                
                payload = {
                    "query_id": f"query_example_{idx:02d}",
                    "domain": domain,
                    "suspicious_claims": [
                        {
                            "claim_id": cid,
                            "text": claim_text
                        }
                    ]
                }
                
                t0 = time.time()
                res = await client.post("/verify", json=payload)
                t1 = time.time()
                
                if res.status_code == 200:
                    data = res.json()
                    reports = data.get("claim_evidence", [])
                    if reports:
                        rep = reports[0]
                        print(f"  • Final Verdict:     {rep.get('verdict')}")
                        print(f"  • Trust Score:       {rep.get('trust_score')}")
                        print(f"  • Support Score:     {rep.get('support_score')}")
                        print(f"  • Contradiction Score: {rep.get('contradiction_score')}")
                        print(f"  • Explanation:       {rep.get('explanation')}")
                        print(f"  • Evidence Count:    {len(rep.get('evidence', []))}")
                        for ev in rep.get('evidence', []):
                            print(f"      - [{ev.get('source')}] {ev.get('title')}")
                            print(f"        Snippet: {ev.get('snippet')[:140]}...")
                            print(f"        Label: {ev.get('entailment_label')} (Score: {ev.get('entailment_score')}, Cred: {ev.get('credibility_score')})")
                    print(f"  • Latency:           {data.get('latency_ms')} ms (Roundtrip: {int((t1-t0)*1000)} ms)")
                else:
                    print(f"ERROR {res.status_code}: {res.text}")
                    
            print("\n" + "=" * 80)
            print("4. ML MODEL MANAGER & INFERENCE VALIDATION")
            print("=" * 80)
            mm = get_model_manager()
            print("Model Loading Status:")
            print(json.dumps(mm.status(), indent=2))
            
            print("\n" + "=" * 80)
            print("5. DOMAIN ADAPTER STATUS MATRIX")
            print("=" * 80)
            registry = get_registry()
            all_domains = registry.list_domains()
            print(f"Total Registered Domains: {len(all_domains)}")
            for d in all_domains:
                adapter = registry.get_adapter(d)
                meta = adapter.metadata
                is_stub = getattr(meta, 'is_stub', False) if hasattr(meta, 'is_stub') else meta.get('is_stub', False)
                status_str = "Stub" if is_stub else "Live / Implemented"
                sources = getattr(meta, 'supported_domains', [])
                print(f"  • Domain: {d:20s} | Sources: {str(sources):30s} | Status: {status_str}")

    print("\n" + "=" * 80)
    print("6. GIT STATUS & REPOSITORY VERIFICATION")
    print("=" * 80)
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    status_out = subprocess.check_output(["git", "status"], cwd=repo_dir, text=True)
    branch_out = subprocess.check_output(["git", "branch"], cwd=repo_dir, text=True)
    log_out = subprocess.check_output(["git", "log", "--oneline", "-5"], cwd=repo_dir, text=True)
    
    print("\n--- git status ---")
    print(status_out)
    print("--- git branch ---")
    print(branch_out)
    print("--- git log --oneline -5 ---")
    print(log_out)

if __name__ == "__main__":
    asyncio.run(main())
