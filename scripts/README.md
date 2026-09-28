# Scripts

The scripts directory contains targeted diagnostics and validation utilities, not product import packages.

| Group | Examples |
|---|---|
| Full demonstrations | `prove_all_agents_e2e.py`, `demo_three_agents_e2e.py` |
| LLM/provider checks | `test_openrouter_llm.py`, `test_llm_detector_slice.py` |
| Retrieval/n8n checks | `test_n8n_retrieval.py`, `test_web_evidence.py`, `repair_n8n_workflow.py` |
| Verifier diagnostics | `diagnose_verifier.py`, `diagnose_verifier_e2e.py`, `verify_claim.py` |
| Certification/readiness | `deployment_readiness_check.py`, `verify_certification_sandbox.py` |
| Documentation | `build_master_doc.py`, `generate_diagrams.py` |

Probe and audit scripts created for one-off debugging should remain local unless they are generalized, credential-safe and documented. Runtime output belongs in ignored log/report paths.
