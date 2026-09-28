# Model Card — HalluciGuard Detector Agent

Last updated: 2026-09 (claim-level hybrid evidence verification edition)

This card documents every model and deterministic component the Detector can
invoke, their intended use, and their known limitations. It follows the honest
diagnostics philosophy of the HalluciGuard hardening effort: **a degraded or
non-calibrated component is labeled as such, never presented as calibrated ML
inference.**

---

## 1. Task

The Detector Agent triages an LLM response for factual-hallucination risk. It
is **not** the final fact checker — the evidence-backed Verifier Agent owns the
final verdict. The Detector answers: *"should this response be accepted on the
fast path, or sent to evidence verification?"*

Two execution modes (see `detector.py`):

1. **Legacy triage** — per-atomic-claim HaluEval classifier + token-surprisal
   signal. No evidence corpus required.
2. **Claim-level hybrid evidence verification** — when an evidence corpus is
   supplied, each claim is typed, rechecked against retrieved evidence, and
   labeled SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO.

---

## 2. Models

### 2.1 HaluEval fine-tuned DistilBERT classifier (legacy path)

- **Base:** `distilbert-base-uncased`, sequence classification head (2 classes: NO_HALLUCINATION / HALLUCINATION).
- **Fine-tuning:** HaluEval public hallucination dataset via `agents/detector_agent/halueval_trainer.py`.
- **Artifact:** `Manjunath2000006/halluciguard-detector` (HF id) or a local directory via `HALUEVAL_MODEL_PATH`.
- **Input:** `format_detector_input(user_query, llm_response)` — query/response text pair.
- **Intended use:** coarse per-claim hallucination-risk signal in the legacy path.

### 2.2 TokenSurprisalEvaluator (HalluDetect-style signal, legacy path)

- Model-based surprisal scoring of the response against the query/claim.
- The legacy path **prefers this signal over the classifier** when it executes,
  because (see 5.3) the checkpoint is collapsed on contextless production claims.

### 2.3 Retrieval backends (evidence path) — reused from the Verifier Agent

- **BM25** (`rank_bm25`, lexical) — `BM25Retriever`, candidates `bm25_k = 10`.
- **Dense** — `DenseRetriever`, `BAAI/bge-m3` embeddings, FAISS `IndexFlatIP`,
  candidates `dense_k = 10`.
- **Reranker** — `CrossEncoderReranker`, `BAAI/bge-reranker-large`, final top-k
  (default `final_k = 3`) from a deduplicated pool (`pool_size = 20`).

### 2.4 Claim-evidence NLI (evidence path)

- `NLIEngine` wrapping `cross-encoder/nli-deberta-v3-base` — 3-way entailment:
  SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO.
- **Design rule:** NOT_ENOUGH_INFO is never treated as CONTRADICTED; the two
  scores are kept disjoint (`nli.py`).

### 2.5 Deterministic fallbacks (no model weights available)

- Lexical BM25-only ordering when dense/rerank backends are unavailable.
- Heuristic claim-evidence classifier (paraphrase lexicon, relation-conflict
  table, evidence-sufficiency guard) when DeBERTa NLI is unavailable.
- These outputs are explicitly **non-calibrated heuristic scores** and always
  set `nli_degraded=true` — they never authorize a `requires_verification=false`
  fast path on their own.

---

## 3. Intended Use

- Apache-2.0-or-compatible research product for hallucination risk triage.
- Fast path (LOW/MEDIUM → Accept) and evidence routing (HIGH → Verify).
- Claim-level evidence verification when the caller can supply a provenance
  corpus (e.g. retrieved documents, grounding context).

## 4. Out of Scope / Misuse

- **Not** a calibrated probability of truthfulness. Never use
  `hallucination_probability` as a precise statistical truth probability.
- **Not** an autonomous fact check: verdicts are confirmed by the Verifier.
- **Not** a domain-specific model: retrieval/typing are domain-agnostic and may
  need tuning for specialized corpora.
- **Not** censorship: the agent can only route requests to verification.

---

## 5. Known Limitations

1. **Non-calibrated scores.** No temperature scaling / Platt scaling exists.
   Classifier, surprisal, rerank, and deterministic scores are routing signals,
   not calibrated probabilities.
2. **Collapsed HaluEval checkpoint.** The supplied checkpoint is documented in
   `detector.py` as "empirically collapsed on contextless production claims";
   the legacy path prefers the surprisal signal and the evidence path replaces
   it entirely.
3. **Evidence mode requires a corpus.** Without `documents`, claims are left
   `UNVERIFIED` (`requires_verification=true`) — fail-closed, never
   hands-free.
4. **Deterministic fallback ≠ NLI.** Fallback labels are heuristic; the
   detector marks `status="degraded"` and refuses the fast path for
   fallback-supported claims.
5. **Opinion heuristic is conservative.** Bare superlatives (e.g. "the best
   X") may be typed OPINION even in objective contexts — a deliberate
   conservative choice to never flag value judgments as factual hallucinations.
6. **Compound-predicate splitting is conservative.** Only passivized
   "was/were VBN … and VBN …" predicates split into separate atomic claims;
   other compound predicates remain single claims (pinned by
   `test_claims.py::test_decompose_preserves_compound_predicate`).

---

## 6. Evaluation

- Contract + integration: `agents/detector_agent/tests` (incl.
  `test_claim_evidence.py` — hermetic, offline, no model downloads).
- Claim decomposition regressions: `agents/verifier_agent/tests/test_claims.py`.
- Detector↔Verifier handoff: `test_detector_verifier_integration.py`.
- Cross-agent suites (verifier, orchestration, memory, judge, corrector) run
  in CI-style regression sweeps; all suites are green with the evidence-mode
  defaults.
- Model-inference-gated tests (`require_model` fixture) run only when the
  HaluEval checkpoint is actually loadable and assert the routing **invariant**
  (LOW/MEDIUM → Accept, HIGH → Verify), not an absolute ACCEPT outcome,
  because the checkpoint's surprisal signal can rate correct answers HIGH.

---

## 7. Intended Runtime Environment

- Python 3.x, `torch`, `transformers`, `sentence-transformers`, `faiss`,
  `rank_bm25`, `spacy` + `en_core_web_sm`.
- Weights resolved from the Hugging Face cache (`~/.cache/huggingface/hub`):
  `Manjunath2000006/halluciguard-detector`,
  `cross-encoder/nli-deberta-v3-base`, `BAAI/bge-reranker-large`,
  `BAAI/bge-m3`, `distilbert-base-uncased`, `Qwen2.5-1.5B-Instruct`.
- Everything degrades softly; no model download is required to keep the
  pipeline fail-closed.

---

## 8. Ethical Considerations

- Routing to verification is a safety measure, not a truth label. A claim
  marked CONTRADICTED by the Detector is a *flag* for the Verifier to confirm,
  and can be overturned by evidence.
- Heuristic fallbacks are clearly surfaced through `degraded` /
  `nli_degraded` / `model_source` fields to prevent silent overconfidence.