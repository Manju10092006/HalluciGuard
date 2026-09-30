"""
Content for the HalluciGuard reviewer PDF (consumed by build_reviewer_pdf.py).

Every technical answer here was checked against the actual repository source before
being written. Block grammar (tuples):
  ("section", title, subtitle, color)
  ("q", num, question_text)
  ("body", html_text)          ("bullets", [html, ...])
  ("formula", latex, caption)  ("callout", title, html, accent_color)
  ("table", headers, rows, col_widths_mm)   ("code", text)
  ("spacer", pts)              ("pagebreak",)
"""
from build_reviewer_pdf import NAVY, TEAL, BLUE, GREEN, RED, AMBER, PURPLE, MUTED

META_ROWS = [
    ["System", "HalluciGuard — a post-generation verification & governance layer around LLM answers"],
    ["Backend", "FastAPI + Python + LangGraph, served by <font face='Courier'>uvicorn orchestration.api:app</font>"],
    ["Hosting", "Next.js frontend on Vercel; Python backend on Render (no Nginx)"],
    ["Base-LLM router", "Groq → Gemini → OpenRouter failover (order configurable)"],
    ["Groq / Gemini / OpenRouter model", "openai/gpt-oss-120b · gemini-flash-latest · qwen/qwen3-14b"],
    ["Detector", "DeBERTa-v3-xsmall 3-class cross-encoder over (evidence, sentence), trained on RAGTruth"],
    ["NLI engine", "cross-encoder/nli-deberta-v3-base (entailment / contradiction / neutral)"],
    ["Reranker / dense embedder", "BAAI/bge-reranker-large · BAAI/bge-m3"],
    ["Deployed retrieval", "Python domain adapters (N8N_RETRIEVAL_ENABLED=false); n8n optional broker"],
    ["Verifier verdicts", "VERIFIED · CONTRADICTED · UNVERIFIED · CONFLICTED"],
    ["Judge actions", "ACCEPT · CORRECT · VERIFY_AGAIN · REJECT · ABSTAIN"],
]

# ---------------------------------------------------------------- key corrections
_INTRO = [
    ("section", "Read this first — corrections a reviewer may probe", "The claims most likely to be challenged, resolved against the code", NAVY),
    ("body", "This dossier is deliberately conservative. The items below are the ones where a "
             "casual description of the system would be <b>wrong</b>, so they are stated correctly up front."),
    ("callout", "The base paper has TWO agents, not five",
     "The referenced paper — <i>Mitigating LLM Hallucinations Using a Multi-Agent Framework</i> "
     "(Darwish, Rashed &amp; Khoriba, MDPI) — uses a two-role <b>consultant / evaluator</b> loop "
     "(LLaMA-3-8B consultant, Mistral-7B evaluator). The multi-stage pipeline is <b>HalluciGuard's own</b> "
     "architecture, not the paper's. Do not claim the paper has five agents.", AMBER),
    ("callout", "No Nginx anywhere",
     "The backend is a Python <b>FastAPI</b> app served by <b>Uvicorn</b> "
     "(<font face='Courier'>uvicorn orchestration.api:app</font>). A repo-wide search finds no Nginx "
     "config or reverse proxy. If asked &ldquo;why Nginx?&rdquo;, the correct answer is that Nginx is not part of the architecture.", RED),
    ("callout", "Legacy 100% benchmark ≠ current full-system accuracy",
     "Older material reports a ~74% / &ldquo;100% precision&rdquo; 35-claim benchmark and a DistilBERT/HaluEval detector. "
     "Those are <b>historical / superseded</b>. The current tracked detector is a DeBERTa-v3-xsmall model trained on "
     "RAGTruth, and its <b>held-out</b> metrics (below) are modest. Never present a legacy score as current overall performance.", RED),
    ("callout", "The Detector is a triage signal, not a truth oracle",
     "The detector&rsquo;s held-out precision (30.8%) and false-negative rate (46.7%) are weak by design for a one-epoch "
     "triage baseline. The model card itself says it must <b>never act autonomously</b> and must always route flagged "
     "claims to the Verifier. Trust in HalluciGuard comes from the evidence-grounded Verifier + Judge, not the Detector.", TEAL),
    ("pagebreak",),
]

# ================================================================ FOUNDATIONS
_FOUND = [
    ("section", "Foundations — why HalluciGuard exists", "Questions 1–10", NAVY),
    ("body", "Questions 1–10 are <b>diagram briefs</b>: each states a reviewer question and the "
             "visual that should answer it. They are reproduced here as the specification for each figure."),

    ("q", 1, "What is an LLM hallucination?"),
    ("body", "<b>Diagram brief.</b> Compare a fluent, confident LLM response with the factual reality. "
             "Show the model producing grammatically perfect, confident text that is unsupported or incorrect, "
             "then draw a clean separation between <i>linguistic fluency</i> and <i>factual correctness</i> — "
             "establishing hallucination as the core problem HalluciGuard addresses."),

    ("q", 2, "Why can a fluent LLM answer still be factually wrong?"),
    ("body", "<b>Diagram brief.</b> Show generation predicting a plausible continuation with no independent "
             "truth check, then two parallel paths: &ldquo;Sounds correct&rdquo; (succeeds on fluency) vs "
             "&ldquo;Actually verified&rdquo; (fails — no external verification occurred). Make it obvious that "
             "confidence and fluency are not factual correctness."),

    ("q", 3, "Why is normal LLM + RAG not enough?"),
    ("body", "<b>Diagram brief.</b> A conventional RAG pipeline (Query → Retrieval → Context → LLM → Answer) "
             "beside HalluciGuard&rsquo;s claim-level verification <i>after</i> generation. RAG can retrieve relevant "
             "context, yet the LLM can still misinterpret, combine, distort or contradict it while writing the answer. "
             "Distinguish <i>retrieving information</i> from <i>independently verifying the generated claims</i>."),

    ("q", 4, "Why is HalluciGuard needed?"),
    ("body", "<b>Diagram brief.</b> Start with Query → Base LLM → Draft, then add the HalluciGuard stages "
             "Detect → Analyze Claims → Retrieve Evidence → Verify → Decide → Correct → Re-verify → Remember. "
             "Communicate that HalluciGuard is a verification-and-governance layer around answers, not another chatbot "
             "or retrieval layer."),

    ("q", 5, "Why use multiple specialized components instead of one LLM?"),
    ("body", "<b>Diagram brief.</b> One general LLM trying to generate, detect, retrieve, verify, decide, correct and "
             "manage memory at once — versus specialized components each owning one task. Show separation of "
             "responsibilities and why each stage needs different safeguards."),

    ("q", 6, "How does the complete HalluciGuard architecture work?"),
    ("body", "<b>Diagram brief.</b> The main end-to-end figure: Query → Base LLM → Draft → Detector → Claim Analyzer "
             "→ Evidence Retrieval / Verifier → Judge → Corrector (when required) → ReVerifier → Final Answer → Memory, "
             "with external services (LLM providers, retrieval sources, optional n8n, frontend/backend boundary) around it."),

    ("q", 7, "How do the agents communicate with each other?"),
    ("body", "<b>Diagram brief.</b> Components connected through a structured shared state / data contract rather than "
             "trading free-form text: execution ID, user query, draft response, claims, evidence, detector results, "
             "verifier results, Judge decision, retry state, trace. They are coordinated components, not isolated chatbots."),

    ("q", 8, "Why can&rsquo;t one component perform another&rsquo;s job?"),
    ("body", "<b>Diagram brief.</b> A responsibility-boundary map: Detector → risk signal; Claim Analyzer → identifies "
             "factual claims; Verifier → evaluates evidence; Judge → chooses workflow action; Corrector → edits authorized "
             "claims; ReVerifier → checks the corrected response. Arrows show each stage depends on the previous stage&rsquo;s "
             "output, so merging them would weaken separation of concerns and make failure analysis harder."),

    ("q", 9, "Does every answer go through Corrector and ReVerifier?"),
    ("body", "<b>Diagram brief.</b> Normal vs conditional paths: after verification the Judge picks ACCEPT, CORRECT, "
             "VERIFY_AGAIN, REJECT or ABSTAIN, and only the CORRECT path proceeds to Corrector → ReVerifier. HalluciGuard "
             "does not rewrite every answer — correction is triggered only when the decision policy requires it."),

    ("q", 10, "Which parts of HalluciGuard are actually implemented?"),
    ("body", "<b>Diagram brief.</b> Separate <b>current production</b> (Detector, Claim Analyzer, Verifier, Judge, "
             "Corrector, ReVerifier, Memory, orchestration, API, frontend/backend, provider router) from <b>optional</b> "
             "(n8n retrieval path) and <b>legacy/experimental</b> (older detector experiments) so nothing theoretical is "
             "shown as the current path."),
    ("callout", "Verified", "Q6/Q10 match the code: the LangGraph pipeline wires generate → detector → "
     "claim_analyzer → verifier → judge → corrector → reverifier → memory in orchestration/graph.py, and n8n is "
     "optional (disabled in the Render deployment).", GREEN),
    ("pagebreak",),
]

# ================================================================ DETECTION
_DETECTION = [
    ("section", "Claim analysis &amp; detection", "Questions 11–20", NAVY),

    ("q", 11, "Why is the Claim Analyzer needed?"),
    ("body", "<b>Diagram brief.</b> A multi-sentence answer enters the Claim Analyzer and is split into individual "
             "atomic claims (one paragraph → several independently verifiable statements). Verifiable claims continue "
             "toward verification; non-verifiable text is filtered or classified separately. HalluciGuard verifies "
             "specific generated claims, not the response as one undifferentiated block."),

    ("q", 12, "How does the Claim Analyzer distinguish factual and non-factual content?"),
    ("body", "<b>Diagram brief.</b> A response containing facts, opinions, instructions, questions, transitions and "
             "metadata is classified span-by-span: FACTUAL_CLAIM continues toward evidence verification, while "
             "OPINION / INSTRUCTION / QUESTION / TRANSITION / META / DISCLAIMER are handled differently. Not every "
             "sentence needs factual verification."),

    ("q", 13, "Why verify generated draft claims instead of the original user query?"),
    ("body", "<b>Diagram brief.</b> A three-stage transformation User Query → LLM Draft → Extracted Claims. The query "
             "is what the user asked; the draft is what the model actually asserted. Verification begins from the "
             "generated claims because hallucinations live in the model&rsquo;s assertions, not in the question."),

    ("q", 14, "Why does the Detector have two phases?"),
    ("body", "<b>Diagram brief.</b> The same response passes two Detector checkpoints. Phase 1 runs <i>before</i> "
             "evidence retrieval as initial risk triage; claim analysis + retrieval then occur; Phase 2 uses the "
             "grounded context for a stronger assessment. Phase 1 cannot replace Phase 2 — before retrieval there is "
             "no external evidence to ground against."),
    ("callout", "Verified — grounding gate", "The code fails <i>closed</i>: when no grounding context is available "
     "(the production Phase-1 case), the detector returns a null probability and routes to the Verifier rather than "
     "emitting a confident score. It never acts autonomously.", GREEN),

    ("q", 15, "Which model does the current Detector use and what does it output?"),
    ("body", "<b>Diagram brief.</b> The current Detector is a <b>DeBERTa-v3-xsmall</b> cross-encoder trained on "
             "<b>RAGTruth</b>, consuming (evidence, sentence) and producing three classes — SUPPORTED, CONTRADICTED, "
             "NOT_ENOUGH_INFO — whose softmax probabilities flow downstream. Do <i>not</i> depict the older "
             "token-probability / entropy / self-consistency detector as the current implementation."),

    ("q", 16, "How was the Detector trained?"),
    ("body", "<b>Diagram brief.</b> RAGTruth → preprocessing → train/dev/test split → DeBERTa-v3-xsmall fine-tuning → "
             "validation → temperature calibration → held-out evaluation, over the three labels, with final metrics on "
             "held-out data (not training examples)."),

    ("q", 17, "How is hallucination probability calculated and calibrated?"),
    ("body", "<b>Diagram brief + formula.</b> The Detector emits three calibrated probabilities; the two risk-oriented "
             "ones combine into the hallucination-risk signal, with temperature scaling applied first. The Detector "
             "produces probabilistic <i>evidence about risk</i>, not a hard yes/no."),
    ("formula", r"P(\text{hallucination}) = P(\text{CONTRADICTED}) + P(\text{NOT\_ENOUGH\_INFO})",
     "Risk signal = the two non-supported class probabilities, after temperature scaling"),
    ("formula", r"T = 0.7586 \qquad \tau \approx 0.6216",
     "Recorded calibration temperature T and decision threshold τ (from the tracked checkpoint)"),

    ("q", 18, "How does sentence-level risk become response-level risk?"),
    ("body", "<b>Diagram brief.</b> Each sentence gets an individual hallucination-risk score; those are aggregated "
             "into a response-level signal. A response with one or more high-risk factual claims can be routed "
             "differently from one whose claims are consistently low-risk."),

    ("q", 19, "What are the actual Detector results?"),
    ("body", "<b>Held-out evaluation</b> on the untouched RAGTruth test split (n = 18,777). These are Detector "
             "held-out metrics, <b>not</b> full-system HalluciGuard accuracy."),
    ("table", ["Metric", "Value", "Metric", "Value"],
     [["Accuracy", "86.55%", "ROC-AUC", "82.12%"],
      ["Precision", "30.83%", "PR-AUC", "28.86%"],
      ["Recall", "53.32%", "ECE", "12.96%"],
      ["F1", "39.07%", "FPR", "10.53%"],
      ["Test set n", "18,777", "FNR", "46.68%"]],
     [42, 43, 42, 43]),
    ("callout", "Verified — exact match", "Every value above matches "
     "artifacts/detector-best/test_metrics.json exactly. The low precision (30.8%) and high false-negative rate "
     "(46.7%) are honest properties of a one-epoch triage baseline — which is why the Detector must always route "
     "flagged claims to the Verifier and never decide alone.", GREEN),

    ("q", 20, "What are the limitations of the Detector?"),
    ("body", "<b>Diagram brief.</b> The Detector is ringed by its limitation categories: dataset dependence, domain "
             "shift, false positives, false negatives, NOT_ENOUGH_INFO ambiguity, calibration limits, and dependence "
             "on available evidence. Detector output is a <i>risk signal, not the final truth decision</i> — so "
             "HalluciGuard combines it with claim analysis, evidence retrieval, Verifier results and Judge policy."),
    ("pagebreak",),
]

# ================================================================ RETRIEVAL
_RETRIEVAL = [
    ("section", "Evidence retrieval &amp; verification", "Questions 21–30", NAVY),

    ("q", 21, "What does n8n actually do?"),
    ("body", "<b>Diagram brief.</b> n8n is an evidence-<i>retrieval / integration</i> orchestration layer, not the "
             "component that decides truth: Claim → Query Builder → Primary Search → Fallback → Extraction → Dedup → "
             "Ranking → Top Evidence → Verifier. The evidence is then handed to the Python Verifier for semantic and "
             "factual evaluation. Keep the boundary between retrieval orchestration and truth verification sharp."),

    ("q", 22, "Is n8n always used for the current production retrieval path?"),
    ("body", "<b>Diagram brief.</b> Two possible retrieval paths — an integrated n8n workflow, and the Python "
             "retrieval-adapter path. The current Render config <b>disables</b> n8n retrieval and uses Python adapters, "
             "so n8n is not mandatory for every production request."),
    ("callout", "Verified", "render.yaml sets <font face='Courier'>N8N_RETRIEVAL_ENABLED=\"false\"</font>. When n8n "
     "is enabled it runs <i>before</i> the Python adapters as a broker; Tavily is an internal quality-gated fallback "
     "inside the web adapter, not a discrete final stage.", GREEN),

    ("q", 23, "How does source routing work?"),
    ("body", "<b>Diagram brief.</b> A claim enters a routing layer that analyzes its domain and information needs and "
             "directs retrieval toward appropriate source categories — different claims route to different evidence "
             "sources rather than every claim using one identical search strategy."),

    ("q", 24, "How are duplicate or copied pieces of evidence handled?"),
    ("body", "<b>Diagram brief.</b> Search results that look different but carry duplicated information pass through a "
             "deduplication / source-diversity stage before ranking. Repeatedly finding the same copied statement is "
             "<i>not</i> independent corroboration from multiple source families."),

    ("q", 25, "How is evidence ranked?"),
    ("body", "<b>Diagram brief.</b> Evidence candidates are scored across relevance, entity match, authority, "
             "coverage, specificity, temporal relevance, domain relevance and source diversity, producing categories "
             "such as STRONG / USABLE / WEAK / DISCARD. The best evidence is not simply the first search result."),

    ("q", 26, "What happens inside the Verifier?"),
    ("body", "<b>Diagram brief.</b> The full internal pipeline: claim validation → decomposition → entity resolution "
             "→ query expansion → domain routing → retrieval → deduplication → relevance ranking → BGE reranking → NLI "
             "→ relation verification → source reliability → conflict resolution → evidence scoring → verdict. The "
             "Verifier is a complete evidence-evaluation system, not a web-search wrapper."),
    ("callout", "Verified — models", "Reranker <font face='Courier'>BAAI/bge-reranker-large</font>; dense embedder "
     "<font face='Courier'>BAAI/bge-m3</font>; NLI <font face='Courier'>cross-encoder/nli-deberta-v3-base</font>; "
     "hybrid BM25 + FAISS fused with Reciprocal Rank Fusion (k = 60).", GREEN),

    ("q", 27, "How does BGE reranking improve evidence selection?"),
    ("body", "<b>Diagram brief.</b> Two stages: a broad retrieval produces many candidates, then the BGE reranker "
             "compares the claim against each candidate and lifts the most semantically relevant passages to the top. "
             "Contrast shallow initial-retrieval relevance with deeper semantic relevance — reranking is why later "
             "verification sees better evidence."),

    ("q", 28, "How do NLI and relation verification work together?"),
    ("body", "<b>Diagram brief.</b> The claim splits into subject → relationship → object alongside the evidence. One "
             "branch runs NLI (supports / contradicts / neutral); the other checks whether the key factual relationship "
             "is actually grounded. The two signals combine before the Verifier concludes — catching entity swaps that "
             "lexical similarity alone would miss."),

    ("q", 29, "What are the possible Verifier outcomes?"),
    ("body", "<b>Diagram brief.</b> One claim branches into VERIFIED, CONTRADICTED, UNVERIFIED, or (where applicable) "
             "CONFLICTED, each tied to the evidence situation that produces it. The verdict is passed to the Judge — it "
             "does not directly become the final answer."),

    ("q", 30, "How are Verifier confidence and trust signals calculated?"),
    ("body", "<b>Diagram brief.</b> The Verifier combines support strength, contradiction strength, evidence count, "
             "source diversity, consensus, source reliability and evidence quality. These feed Verifier "
             "confidence / trust signals — <i>not</i> one universal HalluciGuard &ldquo;Trust Score.&rdquo; The exact "
             "formulas appear in Q63–Q72."),
    ("callout", "Verified — two independent numbers", "Trust Score and calibrated Confidence are computed separately "
     "and mean different things: Trust = how strongly evidence supports the claim (weighted by source reliability); "
     "Confidence = how decisive the available evidence is. Do not present them as one number.", GREEN),
    ("pagebreak",),
]

# ================================================================ JUDGE
_JUDGE = [
    ("section", "Judge &amp; governance", "Questions 31–35", NAVY),

    ("q", 31, "Why is the Judge needed?"),
    ("body", "<b>Diagram brief.</b> Detector outputs and Verifier results arrive at a central Judge positioned as the "
             "<i>workflow-governance</i> layer. It weighs evidence verdicts, contradictions, unverified claims, "
             "confidence, domain/criticality, retry state and policy before choosing an action — separating factual "
             "verification from workflow decision-making."),
    ("callout", "Verified", "In code the Judge is the &ldquo;Chief Decision Officer&rdquo; and explicitly does <i>no</i> "
     "independent fact-checking, NLI inference or refutation — it relies on the Verifier&rsquo;s authoritative "
     "investigation and consumes verifier_result + detector_result.", GREEN),

    ("q", 32, "What states can the Judge produce?"),
    ("body", "<b>Diagram brief.</b> A decision hub with exactly five branches — ACCEPT (release), CORRECT (send to "
             "correction), VERIFY_AGAIN (retry verification), REJECT (refuse release), ABSTAIN (withhold / escalate) — "
             "each wired to its next action. The Judge converts evidence and risk information into one explicit "
             "workflow action."),
    ("callout", "Verified — exactly five", "The authoritative <font face='Courier'>JudgeDecision</font> enum is exactly "
     "{ACCEPT, CORRECT, VERIFY_AGAIN, REJECT, ABSTAIN}. A sixth value (ESCALATE_HUMAN) exists in a separate local enum "
     "but is dead code — never emitted. State the five.", GREEN),

    ("q", 33, "Is the Judge the final factual authority?"),
    ("body", "<b>Diagram brief.</b> A hierarchy: Evidence + Verifier → factual grounding; Judge → workflow decision. "
             "The Judge receives factual results rather than determining truth. It answers &ldquo;what should the "
             "system do next?&rdquo;, while the Verifier answers &ldquo;what does the evidence indicate about this "
             "claim?&rdquo;"),

    ("q", 34, "How does the Judge prevent infinite verification loops?"),
    ("body", "<b>Diagram brief.</b> VERIFY_AGAIN can re-enter verification, but a visible retry counter and maximum "
             "bound the cycle. The configured verification retry budget is <b>2</b>; on exhaustion the system moves to "
             "a safe terminal decision (ABSTAIN / REJECT / escalation) rather than looping. Verification retries are "
             "distinct from correction / reverification attempts."),
    ("callout", "Verified", "<font face='Courier'>max_verification_retries = 2</font>. On exhaustion the no-evidence "
     "and unverified/conflicted paths both fall through to ABSTAIN (a safe withhold), and routing sends "
     "ABSTAIN → human escalation. Note: subsystems differ — the standalone corrector app uses 3, the Phase-2 corrector "
     "package defaults to 2 (hard cap 4); &ldquo;2&rdquo; is the Judge/graph value.", GREEN),

    ("q", 35, "Are Detector, Verifier, and Judge confidence values the same?"),
    ("body", "<b>Diagram brief.</b> Three separate containers: Detector confidence = model probability signal; "
             "Verifier confidence = evidence-grounding strength; Judge decision confidence = confidence in the workflow "
             "decision. Different meanings, different inputs — explicitly <i>not</i> one unified &ldquo;Trust "
             "Score.&rdquo;"),
    ("pagebreak",),
]

# ================================================================ CORRECTION
_CORRECTION = [
    ("section", "Correction &amp; reverification", "Questions 36–41", NAVY),

    ("q", 36, "How does the Corrector know what to change?"),
    ("body", "<b>Diagram brief.</b> The original answer sits beside claim-level verification results; only the claims "
             "the Judge <i>authorized</i> for correction are highlighted. The Corrector receives original response + "
             "authorized targets + verified evidence and operates within an explicit correction boundary — it does not "
             "freely rewrite everything."),

    ("q", 37, "How does selective correction work?"),
    ("body", "<b>Diagram brief.</b> A three-claim answer: the supported claim stays unchanged, the contradicted claim "
             "is replaced using verified evidence, and the unresolved claim is handled by safe policy rather than "
             "invented. The before/after makes targeted correction (not blanket rewriting) obvious."),

    ("q", 38, "How does the Corrector prevent over-correction?"),
    ("body", "<b>Diagram brief.</b> The Corrector is ringed by safety gates — target authorization, original-answer "
             "preservation, number/date/entity preservation, unsupported-addition detection, evidence grounding, "
             "contradiction detection, echo detection, minimal-edit validation, and bounded retries — and a would-be "
             "over-correction is shown being rejected. Correction itself is treated as a risky generation step."),
    ("callout", "Verified — nine gates in the wired package", "All nine safeguards exist as conjunctive validators in "
     "the Phase-2 corrector package the graph actually calls: authorization, original-preservation, number/date/entity "
     "preservation, unsupported-addition, evidence grounding, contradiction, echo (4-rung ladder), minimal-edit (soft), "
     "and bounded retries.", GREEN),

    ("q", 39, "What happens if the Corrector itself produces a wrong answer?"),
    ("body", "<b>Diagram brief.</b> The revised answer is not released directly — it goes straight to independent "
             "validation. Pass → toward release; fail → retry, rejection, or escalation. The message: "
             "&ldquo;correction is not automatically trusted.&rdquo;"),

    ("q", 40, "How does the ReVerifier work?"),
    ("body", "<b>Diagram brief.</b> The corrected response enters a <i>fresh</i> verification pipeline: re-extract "
             "claims, run independent retrieval + verification, check contradictions and positive grounding, apply a "
             "topicality check, and emit PASS or FAIL. Only a passing corrected response reaches release."),
    ("callout", "Verified — fresh &amp; independent + topicality guard", "The production ReVerifier is a graph node "
     "that re-decomposes the corrected text and re-runs the entire verification pipeline (independent retrieval, not "
     "trusting Corrector output). PASS requires COMPLETED + zero remaining contradictions + all re-extracted claims "
     "verified. A query-topicality guard (salient-token coverage, threshold 0.6) can only <i>downgrade</i> a pass — it "
     "never manufactures an accept — blocking true-but-off-topic corrections.", GREEN),

    ("q", 41, "Why not trust the correction automatically?"),
    ("body", "<b>Diagram brief.</b> Fixing one hallucination can introduce a new unsupported statement elsewhere. "
             "Compare Corrector output without verification vs Corrector output followed by independent ReVerification. "
             "An LLM used for correction is still an LLM, so the corrected answer must cross another verification "
             "boundary."),
    ("pagebreak",),
]

# ================================================================ MEMORY
_MEMORY = [
    ("section", "Memory &amp; knowledge", "Questions 42–49", NAVY),

    ("q", 42, "What does the Memory architecture look like?"),
    ("body", "<b>Diagram brief.</b> Three complementary stores — a Knowledge Graph, a FAISS vector store, and a "
             "SQLite verification cache — plus pattern learning and source-trust tracking. Verified information enters "
             "through a controlled write boundary after successful verification / Judge acceptance. Memory is a "
             "structured verification-history system, not a plain text database."),
    ("callout", "Verified — all five", "KG = NetworkX <font face='Courier'>MultiDiGraph</font> (not Neo4j); vector = "
     "FAISS <font face='Courier'>IndexFlatIP</font> over all-MiniLM-L6-v2 (dim 384); cache = aiosqlite "
     "<font face='Courier'>verification_cache</font> (WAL); plus a SQLite PatternLearner and a SQLite "
     "SourceTrustManager.", GREEN),

    ("q", 43, "What exactly is stored in Memory?"),
    ("body", "<b>Diagram brief.</b> A verified claim enters together with its evidence, source, verification result, "
             "timestamp and provenance — a connected record, not just the final sentence. Provenance matters because "
             "future use must retain where a claim came from and why it was accepted."),
    ("callout", "Verified — provenance is split, not one record", "The five fields exist but across subsystems: vector "
     "metadata (domain/verdict/confidence/fact_id/timestamp), KG claim &amp; SOURCE nodes with "
     "<font face='Courier'>MENTIONS</font> edges, and the cache row (claim, verdict, evidence_summary, source_count, "
     "created_at/expires_at). No single row holds all five.", AMBER),

    ("q", 44, "How does HalluciGuard prevent unverified information from entering Memory?"),
    ("body", "<b>Diagram brief.</b> Two paths reach the write boundary: the verified path passes the Verifier + Judge "
             "acceptance gate and enters memory; the unverified / contradicted path is blocked. Duplicate and "
             "contradiction checks sit at the boundary. Core rule: unverified information must not become trusted "
             "memory just because the LLM generated it."),
    ("callout", "Verified — with an important nuance", "The verdict gate is enforced by the <b>orchestrator</b> "
     "(memory writes only when the Judge decision is ACCEPT and reverification passed, filtered to verdict "
     "&lsquo;verified&rsquo;). The memory agent&rsquo;s own store call would accept any verdict, and the duplicate "
     "check blocks a re-store while the contradiction check only <i>alerts</i> (it does not block). So the "
     "&ldquo;only verified is stored&rdquo; property is guaranteed by the orchestrator gate, not the store itself.", AMBER),

    ("q", 45, "What is the Knowledge Graph and why is it needed?"),
    ("body", "<b>Diagram brief.</b> Verified entities are nodes and verified relationships are edges, forming a small "
             "graph rather than a flat text record. Graph structure preserves relationships between facts and provides "
             "structured context for memory operations — useful when the system must understand connections between "
             "entities and claims."),

    ("q", 46, "How is the Knowledge Graph different from vector memory?"),
    ("body", "<b>Diagram brief.</b> Side by side: the KG represents entities and explicit relationships; the FAISS "
             "store represents semantic embeddings and similarity retrieval. On the same claim, the graph answers "
             "relationship-structure questions while vector memory retrieves semantically similar information — "
             "complementary, not interchangeable."),

    ("q", 47, "How does Memory avoid self-reinforcing hallucinations?"),
    ("body", "<b>Diagram brief.</b> A potentially hallucinated claim is blocked at the boundary for lacking verified "
             "evidence; a genuinely verified claim enters with provenance. Duplicate and contradiction detection stop "
             "problematic records from silently reinforcing one another. Memory is downstream of <i>verification</i>, "
             "not downstream of generation."),
    ("pagebreak",),
]

# ---- memory Q48-49 (continued) --------------------------------------------
_MEMORY += [
    ("q", 48, "How does Memory handle outdated information?"),
    ("body", "<b>Diagram brief.</b> Stored information carries timestamps and validity metadata, with mechanisms such "
             "as TTL / cache expiration, graph-edge decay, source-trust decay, contradiction detection, and newer "
             "evidence overriding older where appropriate. Stored information is <i>historical evidence</i>, not "
             "permanent current truth — MEMORY ≠ CURRENT TRUTH."),
    ("callout", "Verified — with corrections", "Real mechanisms: TTL cache expiry (default 24 h, lazy + cleanup); "
     "graph-edge decay (×0.95, <i>manual</i>, no scheduler); source-trust decay (×0.99, <i>manual</i>); contradiction "
     "detection (alert-only). But &ldquo;newer overrides older&rdquo; is <b>not</b> automatic on store — what exists "
     "is recency-weighted <i>reranking at recall</i> (recency = max(0, 1 − age/365)) plus manual update/delete and "
     "cache-key upsert. There is no scheduled decay job and no stale-flagging.", AMBER),

    ("q", 49, "Does Memory automatically feed every future query?"),
    ("body", "<b>Diagram brief.</b> Two distinct concepts: the standalone Memory subsystem <i>has</i> recall "
             "capability, but the current orchestration uses Memory primarily as a <b>write boundary after accepted "
             "verification</b> — it does not auto-inject recalled memory into every new Base LLM request. Do not assume "
             "the production graph feeds memory into every future query."),
    ("callout", "Verified — write boundary, not auto-recall", "The memory node runs only as a terminal step after "
     "Judge ACCEPT (a side-effect / audit layer). The base-LLM node calls generation with the query and conversation "
     "history only — grep finds no memory/recall wiring into generation. Reuse of accepted facts is a documented "
     "design intent, but no recall path is wired into the current generation flow.", GREEN),
    ("pagebreak",),
]

# ================================================================ RESULTS / DEPLOYMENT
_RESULTS = [
    ("section", "Evidence, failure &amp; deployment", "Questions 50–55", NAVY),

    ("q", 50, "What evidence do we actually have that HalluciGuard works?"),
    ("body", "<b>Diagram brief.</b> An evidence hierarchy separating Detector held-out evaluation, component-level "
             "tests, integration / E2E behavior, and historical benchmarks — each labeled with what it actually "
             "proves. The Detector has measured held-out results; a controlled apples-to-apples full-system "
             "hallucination-reduction benchmark is <b>not</b> currently established."),
    ("callout", "Honesty boundary", "Never present the legacy ~74% / &ldquo;100% precision&rdquo; 35-claim benchmark "
     "as current overall HalluciGuard accuracy. The only current, tracked, held-out numbers are the Detector metrics "
     "in Q19.", RED),

    ("q", 51, "What happens when one of the components fails?"),
    ("body", "<b>Diagram brief.</b> A failure-routing map over Detector, Claim Analyzer, retrieval, no-evidence, "
             "contradiction, Corrector, ReVerifier and provider failures — each with its safe fallback (retry, "
             "unverified state, fail-closed, escalation, provider fallback). Reliability is designed in, not assumed "
             "from a successful run."),

    ("q", 52, "What happens if the LLM provider fails?"),
    ("body", "<b>Diagram brief.</b> A provider fallback chain Primary → Secondary → Tertiary → Fail-Closed, with "
             "triggers such as 429, timeout, connection failure, server error and missing API key advancing to the "
             "next provider. If all providers fail the system fails closed — it does not fabricate a response to keep "
             "running."),
    ("callout", "Verified", "Configured order <font face='Courier'>groq → gemini → openrouter</font> (render.yaml), "
     "fail-closed, and a <i>failover</i> chain — not an ensemble vote. Models: groq "
     "<font face='Courier'>openai/gpt-oss-120b</font>, gemini <font face='Courier'>gemini-flash-latest</font>, "
     "openrouter <font face='Courier'>qwen/qwen3-14b</font>.", GREEN),

    ("q", 53, "How do the frontend, backend, and agents communicate?"),
    ("body", "<b>Diagram brief.</b> The full boundary: User → Frontend → FastAPI Backend → Orchestration → Agents → "
             "External Services → Final Response → Frontend. Distinguish HTTP/API communication from internal "
             "orchestration/state, and place Detector, Claim Analyzer, Verifier, Judge, Corrector, ReVerifier and "
             "Memory inside the backend."),

    ("q", 54, "What does the actual deployment architecture look like?"),
    ("body", "<b>Diagram brief.</b> Frontend, backend API, orchestration, LLM providers, verification components, "
             "memory subsystem and retrieval infrastructure — drawn as the <i>actual</i> setup: <b>Vercel</b> frontend, "
             "<b>Render</b> backend, with n8n/ngrok shown only in its real (optional, disabled) role, not as mandatory "
             "for every retrieval request."),
    ("callout", "Verified", "FastAPI + LangGraph served by <font face='Courier'>uvicorn orchestration.api:app</font>; "
     "Vercel frontend, Render backend; <b>no Nginx anywhere</b> in the repository.", GREEN),

    ("q", 55, "How are API keys and sensitive credentials protected?"),
    ("body", "<b>Diagram brief.</b> A clear frontend/backend security boundary: the browser talks to the backend via "
             "API requests, provider keys stay in backend environment variables / secrets, and the backend calls "
             "providers and retrieval services without exposing credentials to the browser. Communicate secret "
             "isolation and backend-controlled access."),
    ("callout", "Note", "In render.yaml provider secrets are marked <font face='Courier'>sync: false</font> (injected "
     "as environment secrets, not committed). Separately, the memory <font face='Courier'>/store</font> endpoint is "
     "unauthenticated by default with CORS <font face='Courier'>*</font> — worth hardening before exposure.", AMBER),
    ("pagebreak",),
]

# ================================================================ E2E + PAPER
_E2E = [
    ("section", "End-to-end demonstrations", "Questions 56–57", NAVY),

    ("q", 56, "Show a complete real hallucination correction example."),
    ("body", "<b>Diagram brief.</b> A full walk-through on the claim &ldquo;The capital of India is "
             "Indianapolis.&rdquo; Base LLM generates the wrong answer → Detector risk → Claim Analyzer extracts the "
             "claim → retrieval finds reliable evidence → Verifier returns CONTRADICTED → Judge selects CORRECT → "
             "Corrector changes it to &ldquo;New Delhi&rdquo; → ReVerifier independently confirms → released and stored "
             "as verified memory. The single strongest demonstration of the whole pipeline."),

    ("q", 57, "How does HalluciGuard handle a complex answer containing multiple claims?"),
    ("body", "<b>Diagram brief.</b> One answer with ≥ 3 factual claims — one SUPPORTED, one CONTRADICTED, one "
             "INSUFFICIENT/UNVERIFIED — processed independently with different actions rather than judging the whole "
             "response correct or incorrect. Selective handling per claim, based on that claim&rsquo;s evidence."),
    ("pagebreak",),
]

_PAPER = [
    ("section", "Base paper &amp; research positioning", "Questions 58–60", NAVY),

    ("q", 58, "How does HalluciGuard differ from the base paper?"),
    ("body", "<b>Diagram brief.</b> A side-by-side: on the paper side, its verified two-role structure and correction "
             "loop; on the HalluciGuard side, the implemented pipeline (claim analysis, Detector, retrieval, Verifier, "
             "Judge, Corrector, ReVerifier, Memory). Mark which components are inherited concepts, which are "
             "extensions, and which are additional engineering. Invent no paper details or performance comparisons."),
    ("callout", "Verified — two agents, not five", "The base paper (Darwish, Rashed &amp; Khoriba, MDPI — "
     "<i>Mitigating LLM Hallucinations Using a Multi-Agent Framework</i>) uses a two-role consultant / evaluator loop "
     "(LLaMA-3-8B consultant, Mistral-7B evaluator). The multi-stage pipeline is HalluciGuard&rsquo;s own "
     "architecture.", AMBER),

    ("q", 59, "What quantitative results can we actually claim?"),
    ("body", "<b>Diagram brief.</b> A claim-boundary visual with three bins: <b>measured &amp; supported</b> (the "
             "Detector held-out metrics from Q19), <b>historical / component-specific</b> (clearly labeled legacy "
             "benchmarks, not current), and <b>not yet established</b> (full-system hallucination reduction, "
             "apples-to-apples comparison vs the paper, production-scale latency). Show research honesty; prevent "
             "overclaiming."),

    ("q", 60, "What is the complete trust architecture of HalluciGuard?"),
    ("body", "<b>Diagram brief.</b> The executive summary as a sequence of trust boundaries: Generation → Claim "
             "Extraction → Risk Detection → Evidence Acquisition → Evidence Verification → Governance Decision → "
             "Selective Correction → Independent ReVerification → Verified Memory. Under each stage, what it "
             "contributes and what it cannot be trusted to do alone."),
    ("bullets", [
        "LLM generation is <b>not</b> truth.",
        "Detector risk is <b>not</b> factual proof.",
        "Retrieval is <b>not</b> truth.",
        "The Judge is <b>not</b> the factual authority.",
        "Correction is <b>not</b> automatically trustworthy.",
        "Memory is <b>not</b> current truth.",
    ]),
    ("body", "Trust is built through multiple independent stages, not through a single model."),
    ("pagebreak",),
]

# ================================================================ ANSWERED (Q61-86)
_ANSWERS = [
    ("section", "Answered questions — formulas &amp; deep detail", "Questions 61–86 (full text answers)", NAVY),
    ("body", "Q61–86 carry full written answers in the source document. Each formula below is reproduced exactly as "
             "implemented in the repository; verified nuances a reviewer may probe are called out in colour."),

    ("q", 61, "What exact formula does the Detector use to calculate hallucination probability?"),
    ("body", "The current production Detector is a three-class DeBERTa-v3-xsmall cross-encoder over "
             "<font face='Courier'>(evidence, sentence)</font>, producing P(SUPPORTED), P(CONTRADICTED) and "
             "P(NOT_ENOUGH_INFO). The hallucination-risk probability is the sum of the two non-supported classes:"),
    ("formula", r"P(\text{hallucination}) = P(\text{CONTRADICTED}) + P(\text{NOT\_ENOUGH\_INFO})",
     "Calibrated with temperature scaling (T = 0.7586); recorded decision threshold ≈ 0.6216"),
    ("callout", "Do not mix detector lineages", "The <i>older</i> detector docs contain token-probability, entropy, "
     "semantic-similarity and self-consistency formulas. Those belong to a superseded architecture — not the current "
     "DeBERTa production artifact. This is also not a raw logit score, and the Detector cannot determine truth without "
     "evidence.", AMBER),

    ("q", 62, "Why a multi-stage architecture when the paper uses only two agents?"),
    ("body", "The premise is reversed: the <b>paper</b> uses two roles per sub-task — a consultant (generates/refines) "
             "and an evaluator (runs the rule-based evaluation), repeated inside its nested-task structure. "
             "HalluciGuard goes further because its problem is broader, separating the trust boundaries "
             "Generate → Detect → Analyze → Retrieve → Verify → Judge → Correct → ReVerify → Memory."),
    ("bullets", [
        "Detector estimates risk; Claim Analyzer decides what is checkable.",
        "Verifier obtains and evaluates external evidence; Judge decides the workflow action.",
        "Corrector performs bounded repair; ReVerifier independently validates it.",
        "Memory persists only accepted verified information.",
    ]),
    ("body", "The design principle: a model should not generate a statement and then certify its own statement as true."),

    ("q", 63, "What is the Trust Score formula in the Verifier?"),
    ("body", "For each evidence item the Verifier computes a base weight:"),
    ("formula", r"W = C \times R \times Rel \times V",
     "C = source credibility, R = recency, Rel = relevance weight, V = NLI validity"),
    ("formula", r"Rel = \max(0.20,\ \min(1.0,\ s^{0.25}))",
     "Relevance transform of the BGE / cross-encoder score s (with a relevance floor)"),
    ("body", "Support and contradiction scores aggregate the max and mean weights plus a capped source-diversity bonus:"),
    ("formula", r"S_{support} = \min\!\left(1,\ 0.70\,W_{max} + 0.30\,\overline{W} + Bonus\right)",
     "Contradiction score is identical over contradicting evidence"),
    ("formula", r"Bonus = \min(0.15,\ 0.05\,(n-1))",
     "n = number of distinct supporting / contradicting sources"),
    ("formula", r"T = S_{support}\,(1 - 0.90\,S_{contradiction}) + 0.05\,\min\!\left(1,\ \tfrac{N_{sources}}{2}\right)",
     "Trust Score — only when S_support > S_contradiction AND S_support ≥ 0.25; otherwise T = 0"),
    ("callout", "Verified — base vs effective weight", "W above is the <b>base</b> weight. The effective weight is "
     "further multiplied by an NLI signal, entailment × (1 − 0.35 × neutral). The coefficients (0.70 / 0.30 / 0.90 / "
     "0.05, s^0.25, bonus cap 0.15, gate 0.25) all match the implementation exactly.", GREEN),
    ("pagebreak",),

    ("q", 64, "How do we know the Trust Score is not an arbitrary number?"),
    ("body", "Because it is built from measurable evidence properties, not an LLM asserting &ldquo;I think this is "
             "true.&rdquo; The Verifier combines evidence relevance, NLI entailment/contradiction, source credibility, "
             "publication recency, NLI validity, independent-source diversity, relation/entity consistency and "
             "URL-level deduplication — and keeps Trust Score separate from calibrated Confidence."),
    ("callout", "Trust vs Confidence", "Trust asks how strongly the evidence supports the claim, weighted by source "
     "reliability. Confidence asks how decisive the available evidence is. They are documented and computed as "
     "independent numbers.", BLUE),

    ("q", 65, "What happens if the Verifier cannot find enough evidence?"),
    ("body", "It does not convert &ldquo;no evidence&rdquo; into &ldquo;false.&rdquo; Insufficient evidence yields the "
             "verdict UNVERIFIED. The four verdicts are VERIFIED, CONTRADICTED, UNVERIFIED and CONFLICTED; "
             "NOT_ENOUGH_INFO at the Detector means the supplied evidence is insufficient, not that the claim is "
             "globally false."),
    ("callout", "Reviewer soundbite", "&ldquo;Absence of evidence is treated as uncertainty, not as evidence of "
     "falsehood.&rdquo;", TEAL),

    ("q", 66, "How does the Judge prevent infinite loops?"),
    ("body", "The Judge operates with bounded retry / correction loops. Its decisions are ACCEPT, CORRECT, "
             "VERIFY_AGAIN, REJECT and ABSTAIN. VERIFY_AGAIN returns to verification, but the orchestration keeps "
             "retry/correction state and enforces bounded limits — if the claim cannot be resolved within the allowed "
             "workflow, it moves toward rejection, abstention or human review rather than looping forever."),
    ("callout", "Verified", "<font face='Courier'>max_verification_retries = 2</font>; on exhaustion → ABSTAIN → human "
     "escalation. This differs from the paper&rsquo;s consultant-evaluator loop, which stops after reaching zero "
     "mistakes or after its second iteration.", GREEN),

    ("q", 67, "How does selective correction stop the Corrector changing the whole answer?"),
    ("body", "The Corrector receives the original response, the Judge-authorized claims, and the verified evidence, "
             "and is designed for targeted evidence-bound repair: verified claims are preserved, only authorized "
             "incorrect claims are changed, unsupported facts cannot be invented, and the ReVerifier independently "
             "checks the result. The flow is Incorrect claim → Evidence → Corrector → ReVerifier, never Incorrect "
             "answer → LLM rewrite → trust it."),

    ("q", 68, "Which datasets are actually used — and what about TruthfulQA / HotpotQA?"),
    ("body", "The current production Detector artifact is trained and evaluated on <b>RAGTruth</b> human annotations, "
             "with three classes (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO). Recorded split:"),
    ("table", ["Split", "Examples"],
     [["Train", "29,832"], ["Development", "10,873"], ["Untouched test", "18,777"]],
     [60, 40]),
    ("callout", "Safe answer if asked about TruthfulQA/HotpotQA", "&ldquo;No — the current production Detector is "
     "trained and evaluated on RAGTruth. TruthfulQA / HaluEval and others appear in the repo&rsquo;s "
     "evaluation/research infrastructure, but are not the training source of the current checkpoint unless a specific "
     "recorded experiment proves it.&rdquo;", AMBER),
    ("pagebreak",),
]

_ANSWERS += [
    ("q", 69, "What is the evidence relevance formula?"),
    ("body", "The Verifier converts the reranker score s into a bounded relevance weight, keeping tiny cross-encoder "
             "scores from eliminating evidence while preserving a relevance floor:"),
    ("formula", r"R_w = \max(0.20,\ \min(1.0,\ s^{0.25}))", "Bounded relevance weight"),

    ("q", 70, "What is the base evidence weight?"),
    ("formula", r"W_{base} = Credibility \times Recency \times Relevance \times Validity",
     "Evidence is not trusted merely for being semantically similar"),

    ("q", 71, "How is source recency calculated?"),
    ("body", "The Verifier&rsquo;s source-reliability manager uses a linear decay with a floor:"),
    ("formula", r"R_{recency} = \max(0.6,\ 1 - 0.05 \times YearsSincePublication)",
     "Verifier source-recency weighting"),
    ("body", "The Judge-side evidence-intelligence engine uses a separate exponential half-life:"),
    ("formula", r"S_{fresh} = e^{-\ln(2)\,\cdot\, age / half\_life}",
     "Judge evidence-intelligence freshness (domain-specific half-life, default 365 d)"),
    ("callout", "Verified — two distinct formulas", "Both exist and are different: the linear "
     "max(0.6, 1−0.05·yrs) is the Verifier source weight; the exponential half-life form is the Judge&rsquo;s "
     "EvidenceIntelligenceEngine. Present them separately, not as one.", GREEN),
    ("pagebreak",),

    ("q", 72, "What is the calibrated Confidence Score formula?"),
    ("body", "The implementation identifies the primary score, a consensus factor and a count factor, then multiplies:"),
    ("formula", r"P_{primary} = \max(S_{support},\ S_{contradiction})", ""),
    ("formula", r"ConsensusFactor = \max\!\left(0.10,\ 1 - \min(S_{support},\ S_{contradiction})\right)", ""),
    ("formula", r"CountFactor = 0.75 + 0.25\,\min\!\left(1,\ \tfrac{N_{verified}}{3}\right)", ""),
    ("formula", r"Confidence = P_{primary} \times ConsensusFactor \times CountFactor",
     "Bounded to [0, 1]; deliberately independent of Trust Score"),
    ("callout", "Verified — unstated gates", "Two gates the source text omits but the code applies: if "
     "P_primary &lt; 0.25 the confidence is forced to 0, and an ungrounded VERIFIED verdict is capped at 0.70. "
     "Worth knowing if a reviewer probes edge cases.", GREEN),

    ("q", 73, "What is the RRF formula used in retrieval?"),
    ("body", "The Verifier fuses ranked lists from multiple retrieval methods with Reciprocal Rank Fusion:"),
    ("formula", r"RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}",
     "d = document, M = retrieval methods, r_m(d) = rank of d under method m, k = 60"),
    ("body", "HalluciGuard uses hybrid retrieval — BM25 sparse + FAISS dense — followed by BGE reranking."),

    ("q", 74, "How does NLI calculate evidence support?"),
    ("body", "The NLI model <font face='Courier'>cross-encoder/nli-deberta-v3-base</font> produces three semantic "
             "probabilities — P(E) entailment, P(C) contradiction, P(N) neutral. The Verifier does not blindly take "
             "the largest NLI probability; it combines NLI with relevance gates, relation verification, source "
             "credibility and recency."),

    ("q", 75, "What is the relation-verification logic?"),
    ("body", "Not a single continuous formula — the repository implements SVO / entity-relation checks. A claim like "
             "&ldquo;A was founded by B&rdquo; decomposes conceptually into Subject = A, Relation = FoundedBy, "
             "Object = B. If the evidence establishes a different object/entity, the relation verifier can override an "
             "otherwise misleading NLI result and mark the passage contradictory — specifically to catch entity swaps "
             "that lexical similarity misses."),
    ("pagebreak",),
]

_ANSWERS += [
    ("q", 76, "How does the Verifier decide which website/source is best?"),
    ("body", "It does not simply pick the first search result. The pipeline is approximately Claims → Domain Routing → "
             "Multi-source Retrieval → Deduplication → BM25/FAISS → BGE Reranking → NLI → Relation Checks → Source "
             "Credibility → Recency → Evidence Aggregation. A source must be relevant, sufficiently authoritative, "
             "semantically supportive or contradictory, appropriately current, and not merely duplicated."),
    ("callout", "Verified", "The Python Verifier owns ranking, NLI, evidence semantics and verdicts. n8n is only a "
     "retrieval broker when enabled.", GREEN),

    ("q", 77, "Does the system treat Wikipedia and PubMed equally?"),
    ("body", "No. The source-reliability system classifies sources into authority tiers, for example:"),
    ("bullets", [
        "PubMed / NIH / FDA / WHO / government → high-authority tier",
        "peer-reviewed sources → high academic tier",
        "enterprise / vendor documentation → separate tier",
        "reputable news → below official / academic",
        "Wikipedia / GitHub / StackOverflow / community → community tier",
        "unknown sources → unverified",
    ]),
    ("callout", "Verified — tiers are numeric", "The &ldquo;tiers&rdquo; are concrete credibility <i>floats</i> in the "
     "repo config (web_retriever TRUSTED_DOMAINS + domain_intelligence.yaml), not named enums. Extra domain sources "
     "exist too — e.g. Healthcare + WHO, Cyber + GitHub advisories / CIRCL. Semantic relevance alone does not make a "
     "source authoritative.", GREEN),

    ("q", 78, "What happens if the highest-ranked website is actually bad?"),
    ("body", "The system does not stop because one source ranked first. Multiple candidates are retrieved, "
             "deduplicated, reranked, passed through NLI, checked for relation consistency, weighted by source "
             "credibility and recency, and aggregated — so a highly ranked but weak or contradictory source can be "
             "outweighed by stronger evidence."),

    ("q", 79, "What happens when different websites disagree?"),
    ("body", "HalluciGuard supports CONFLICTED as a separate verdict. It does not force &ldquo;Website A says X, "
             "therefore X.&rdquo; It evaluates competing support and contradiction signals; if both sides are "
             "sufficiently strong and close, the claim becomes CONFLICTED — which is different from having no evidence."),

    ("q", 80, "What is the fallback if the primary retrieval source fails?"),
    ("body", "The architecture layers retrieval: primary/domain adapters → n8n broker (when enabled) → Python "
             "adapters / web fallback → optional Tavily fallback. The current Render deployment sets "
             "<font face='Courier'>N8N_RETRIEVAL_ENABLED=false</font>, so it uses the Python adapters rather than "
             "requiring n8n."),
    ("callout", "Verified — ordering nuance", "When enabled, n8n runs <i>before</i> the Python adapters as a broker. "
     "Tavily is an internal quality-gated fallback <i>inside</i> the web-enhanced adapter, not a discrete final "
     "stage.", GREEN),

    ("q", 81, "What happens if there is no evidence at all?"),
    ("body", "The system fails safely: No-evidence ⇒ UNVERIFIED ⇒ no automatic factual certification. It never does "
             "No-evidence ⇒ False. The Judge can then route toward ABSTAIN, REJECT, retry or human review depending on "
             "state and policy."),

    ("q", 82, "What structured / tabular data sources does the Verifier use?"),
    ("body", "The repo has domain-specific adapters rather than only generic web pages, because APIs provide "
             "structured fields and provenance:"),
    ("table", ["Domain", "Sources"],
     [["Healthcare", "PubMed, PubMed Central, openFDA, ClinicalTrials.gov"],
      ["Cybersecurity", "NVD CVE API, MITRE ATT&CK, CISA KEV"],
      ["Finance", "SEC EDGAR, World Bank, Alpha Vantage"],
      ["AI research", "arXiv, Semantic Scholar, CrossRef"],
      ["Legal / general", "CourtListener, Wikipedia"]],
     [40, 130]),
    ("pagebreak",),
]

_ANSWERS += [
    ("q", 83, "Why do we use multiple LLM providers?"),
    ("body", "The repository implements a configurable provider failover chain:"),
    ("formula", r"Groq \rightarrow Gemini \rightarrow OpenRouter",
     "Order configurable; models gpt-oss-120b · gemini-flash-latest · qwen/qwen3-14b"),
    ("body", "On a retryable failure (transport/server error, unavailable model, rate limit) the system advances to "
             "the next provider. This is a <b>reliability</b> mechanism, not an ensemble where three models vote on "
             "truth."),

    ("q", 84, "Why not use the Base LLM itself as the Verifier?"),
    ("body", "Because that violates the trust boundary. The Base LLM generates the candidate answer; the Verifier must "
             "independently evaluate evidence. Asking the same generator &ldquo;did your answer hallucinate?&rdquo; "
             "lets the generator certify its own output — which HalluciGuard intentionally avoids, and the repository "
             "states this design principle explicitly."),

    ("q", 85, "What is the role of n8n versus Python?"),
    ("body", "n8n is an integration / retrieval <b>broker</b> (when enabled): it can call providers, orchestrate HTTP "
             "workflows, normalize returned data and provide retrieval traces. The Python Verifier owns the "
             "verification intelligence — ranking, relevance, NLI, relation checking, evidence scoring, source "
             "weighting and verdict generation."),
    ("formula", r"n8n \neq Truth\ Authority", "The Python Verifier is the factual decision boundary"),

    ("q", 86, "Why did we choose Nginx over Python?"),
    ("body", "<b>Do not answer that we chose Nginx.</b> A repository-wide search finds no Nginx implementation or "
             "configuration in the project. The actual backend is:"),
    ("formula", r"FastAPI + Python + LangGraph", "served by  uvicorn orchestration.api:app"),
    ("body", "The frontend is deployed through Vercel; the Python backend is configured for Render. So if a reviewer "
             "asks &ldquo;Why Nginx?&rdquo;, the accurate answer is: <b>&ldquo;Nginx is not part of our current "
             "deployed architecture. Our backend is a Python FastAPI service served through Uvicorn. If we later need "
             "a reverse proxy, TLS termination or static-asset serving in front of it, Nginx would be a reasonable "
             "addition — but it is not what the system uses today.&rdquo;</b>"),
    ("callout", "Verified", "No Nginx config or reverse proxy anywhere in the repo. Start command "
     "<font face='Courier'>uvicorn orchestration.api:app</font>; Vercel frontend + Render backend confirmed in "
     "render.yaml.", GREEN),
]

# ================================================================ ASSEMBLY
CONTENT = (
    _INTRO
    + _FOUND
    + _DETECTION
    + _RETRIEVAL
    + _JUDGE
    + _CORRECTION
    + _MEMORY
    + _RESULTS
    + _E2E
    + _PAPER
    + _ANSWERS
)

