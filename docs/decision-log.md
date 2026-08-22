# HalluciGuard Architecture Decision Log (ADR)

## ADR-001: Decoupling Adapter Prior Confidence from Relevance Scoring
* **Context**: Adapters previously returned hard-coded `relevance_score` fields (`0.85` for Wikipedia, `0.90` for NVD). The quality gate evaluated these constants before BGE reranking, causing inappropriate skipping of web fallback search.
* **Decision**: Rename adapter priors to `source_confidence_hint` and evaluate pre-ranking quality using dynamic lexical overlap and query-term coverage ($overlap\_ratio \times 0.7 + hint \times 0.3$).
* **Consequences**: Quality gate accurately detects off-topic primary passages and triggers Tavily fallback only when needed.

---

## ADR-002: MediaWiki Deep Section Retrieval
* **Context**: Querying only the REST summary endpoint (`/page/summary/{title}`) missed critical factual details located in body sections (such as parentage in *"Early life and family"* or historical origin sections).
* **Decision**: Implement Action API extraction (`explaintext=1`) with line-by-line section header parsing and sentence-level window chunking, prioritized by query token overlap.
* **Consequences**: Deep factual relationships are surfaced to the reranker and NLI models with minimal latency increase (~80-120ms).

---

## ADR-003: Deterministic Relation Verification & Coverage Suppression Bypass
* **Context**: Legacy contradiction suppression required 100% claim-word coverage in the snippet. For false claims (e.g. *"The Eiffel Tower is in London"*), refuting evidence (*"The Eiffel Tower is in Paris"*) never contains the false token (*"London"*), causing true contradictions to degrade into `UNVERIFIED`.
* **Decision**: Introduce `RelationVerifier` (extracting SVO triples for capital, location, kinship, creation/invention, vulnerability, and typing) and explicitly **bypass** word-coverage suppression when `OBJECT_MISMATCH` or `RELATION_MISMATCH` is confirmed.
* **Consequences**: Eliminates the false-token containment vulnerability. False-attribution claims correctly resolve to `CONTRADICTED`.

---

## ADR-004: Strict 4-Verdict Public Contract
* **Context**: Inconsistent legacy aliases (`LIKELY_HALLUCINATED`, `INSUFFICIENT_EVIDENCE`, `MIXED_EVIDENCE`) existed across various components.
* **Decision**: Enforce strictly 4 public verdict values (`VERIFIED`, `CONTRADICTED`, `UNVERIFIED`, `CONFLICTED`) and 4 evidence classes (`SUPPORTING`, `CONTRADICTING`, `NEUTRAL`, `IRRELEVANT`).
* **Consequences**: Seamless end-to-end typing across API, CLI, frontend, and tests.

---

## ADR-005: Honest Confidence Calibration
* **Context**: Confidence scores previously mirrored adapter credibility priors, over-claiming confidence when evidence was weak.
* **Decision**: Base confidence strictly on decision-grade evidence strength, degree of agreement, and conflict penalties.
* **Consequences**: Transparent, calibrated confidence scores that honestly reflect model certainty.

---

## ADR-006: Bidirectional Relational Query Expansion
* **Context**: Repeating the exact natural language claim as the only search query often yielded poor search recall for inverted relations (e.g. searching *"Ram Charan invented Java"* on Wikipedia failed to surface James Gosling).
* **Decision**: Generate both active and passive relational query variants (*"who created Java"*, *"Java creator"*, *"Java invented by"*).
* **Consequences**: Drastically improves primary retrieval recall for counterfactual and false-attribution claims.
