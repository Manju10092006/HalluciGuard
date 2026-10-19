# Seven-agent validation — 2026-10-19

Real correction trace: `runtime-logs/local-correction/raw/SUPPLIED_DRAFT_RUN_execution.json`. No verdicts/evidence/providers were mocked. Observer wrappers record original methods in the real compiled LangGraph. Supplied draft bypasses initial generation only.

| Agent | Actual state boundary / observed responsibility |
|---|---|
| Detector | Pre-retrieval claims have no truth probability; grounded pass loads the actual checkpoint and scores retrieved evidence. `model_loaded=true`, `inference_executed=true` observed. |
| Claim Analyzer | Whole draft → typed factual claims with IDs/context/searches; deterministic source coverage backstop; hosted extraction cannot silently omit another source assertion. |
| Verifier | Submitted claim IDs/text → actual retrieved passages, NLI/relation verdicts and canonical reports. Initial incomplete/duplicate/unexpected/mismatched coverage now fails closed. Configured n8n fails; Python retrieval fallback actually ran. |
| Judge | Evidence-backed verified claims → ACCEPT; any unsupported/peripheral remainder → retry/ABSTAIN; contradictions → bounded correction; exceptions/degraded evidence never imply verification. Bare decisive verdicts lack authority. |
| Corrector | Actual contradicted claim + evidence → untrusted corrected candidate. Live provider `openrouter` observed. No candidate becomes verified merely by being returned. |
| ReVerifier | Re-extracts candidate, retains deterministic source assertions, independently retrieves/verifies all corrected claims; requires complete matching IDs, supported evidence and completed verification. Judge independently rejects a passed flag without reports/evidence. |
| Memory | Every terminal route audits eligibility; only ACCEPT + eligible evidence-backed reports persist. Corrections require successful reverification. Unknown/pending/unsupported write requests and verified flags without provenance are rejected. Explicit contradicted/likely_hallucinated records remain negative audit/pattern records, not verified facts. |

Observed live route: generate(supplied) → detector → claim_analyzer → verifier → grounded_detector → judge(CORRECT) → corrector → reverifier → judge(ACCEPT) → memory. Three final verified claims stored in fresh diagnostic files. Unsupported fictional revenue reached human review and Memory skipped with zero writes. Query-only France-capital generation reached ACCEPT and stored one claim.

Initial live traces predate additional safety hardening. The overlapping runtime-logs/final-correction run was stopped under memory pressure and is not a passing result. Final-source HTTP traces are retained under runtime-logs/http-final; the initial negative-only repair was rejected safely, and a separate supported repair exposed and reproduced a false topicality rejection. That defect was fixed without changing the coverage threshold. Deterministic regression suites separately cover timeout, node failure, failed correction, omitted/new claims, mixed evidence and Memory rejection; those are fixture-based, not live-provider failure certification.

Memory recall is tested through its real standalone API/class lifecycle tests; the current LangGraph does not retrieve Memory facts into generation or use Memory as factual proof. Graph Memory is a terminal persistence/audit stage.

Final-source live acceptance after the topicality fix: **PASS** (correction-after-fix.log, runtime-logs/http-final/positive-correction-after-fix.json). HTTP 200 in 129.0 seconds; final answer "Microsoft was founded by Bill Gates and Paul Allen."; terminal accepted / verified_and_accepted; independent reverification passed; all seven trust agents appear in the actual trace; two verified facts stored in a new isolated runtime directory and reopened with attributed evidence. The local frontend responded HTTP 200. Earlier negative-only and false-topicality failures remain preserved.

Local delivery: frontend http://localhost:3020/chat and actual backend http://127.0.0.1:8020, using runtime-logs/api-runtime-delivery. The launcher uses the original provider configuration and process-only local auth key. Existing production Memory/user files remain unchanged and are not retroactively certified or migrated.
