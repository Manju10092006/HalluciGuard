"""
HalluciGuard Verifier Agent — 9-Stage Verification Pipeline Orchestrator.

Runtime flow:
  Claim → Domain Validation → Claim Decomposition → Entity Resolution & Query Expansion →
  Domain Router → Load Correct Models → Generate Domain Embeddings →
  Dense Retrieval + Sparse Retrieval → Hybrid Retrieval → Evidence Aggregation →
  Cross Encoder Reranking → Natural Language Inference → Evidence Fusion →
  Source Reliability → Trust Score → Confidence Calibration →
  Citation Generation → Structured Response
"""

from __future__ import annotations

import uuid
import time
import logging
import math
import hashlib
from typing import List, Dict, Any, Tuple

from schemas.models import (
    VerifierInputV2,
    VerifierOutputV2,
    ClaimReport,
    EvidenceItem,
    SuspiciousClaim,
    Passage,
    EntailmentLabel,
    VerdictLabel,
    PipelineStage,
    PipelineStageStatus,
    RuntimeModelInfo,
)
from adapters.registry import get_registry
from claims import ClaimDecomposer, ClaimNormalizer, ClaimMerger, EntityResolver
from retrievers import HybridRetriever
from aggregation import EvidenceAggregator
from rerankers import CrossEncoderReranker
from nli import NLIEngine
from scorers import EvidenceScorer, SourceReliabilityManager, ConflictResolver
from explanations import ExplanationGenerator
from formatters import CitationFormatter, ResponseFormatter
from routers import DomainValidator, QueryExpander
from models import ModelRouter
from cache import SqliteCache
from metrics import MetricsCollector, PerformanceTracker
from utils.logging import setup_logger
from api.certification import CertificationError


def append_retrieval_health_stage(
    pipeline_stages: List[PipelineStageStatus],
    adapter_failures: List[str],
    total_retrieved: int,
) -> None:
    """Record retrieval health without invalidating a successful fallback.

    A provider failure is terminal only when the retrieval chain produced no
    passages.  If a later adapter recovered and returned evidence, the
    retrieval stage completed authoritatively; the failed provider remains in
    ``details`` for observability but must not force the Judge to abstain.
    """
    if adapter_failures and total_retrieved == 0:
        pipeline_stages.append(
            PipelineStageStatus(
                stage=PipelineStage.RETRIEVAL,
                status="failed",
                duration_ms=0,
                details=f"All retrieval paths failed: {', '.join(adapter_failures)}",
            )
        )
    elif adapter_failures:
        pipeline_stages.append(
            PipelineStageStatus(
                stage=PipelineStage.RETRIEVAL,
                status="completed",
                duration_ms=0,
                details=(
                    "Fallback retrieval succeeded after provider failure: "
                    f"{', '.join(adapter_failures)}"
                ),
            )
        )
    elif total_retrieved == 0:
        pipeline_stages.append(
            PipelineStageStatus(
                stage=PipelineStage.RETRIEVAL,
                status="degraded",
                duration_ms=0,
                details=(
                    "No passages were retrieved; downstream reranking, NLI, "
                    "and evidence scoring had no real evidence to process."
                ),
            )
        )


class VerificationPipeline:
    """
    The 9-stage verification pipeline orchestrator.

    Integrates entity resolution, domain-specific model routing, runtime model metadata,
    confidence calibration, and cache-hit indicators into every response.
    """

    def __init__(self) -> None:
        self.claim_decomposer = ClaimDecomposer()
        self.claim_normalizer = ClaimNormalizer()
        self.claim_merger = ClaimMerger()
        self.entity_resolver = EntityResolver()
        self.hybrid_retriever = HybridRetriever()
        self.aggregator = EvidenceAggregator()
        self.reranker = CrossEncoderReranker()
        self.nli_engine = NLIEngine()
        self.evidence_scorer = EvidenceScorer()
        self.reliability_manager = SourceReliabilityManager()
        self.conflict_resolver = ConflictResolver()
        self.explanation_generator = ExplanationGenerator()
        self.citation_formatter = CitationFormatter(evidence_scorer=self.evidence_scorer)
        self.response_formatter = ResponseFormatter()
        self.domain_validator = DomainValidator()
        self.model_router = ModelRouter()
        self.query_expander = QueryExpander()
        self.cache = SqliteCache()
        self.metrics = MetricsCollector()
        self.logger = setup_logger("pipeline")
        from agents.verifier_agent.config.settings import get_settings
        self.settings = get_settings()
        self._n8n_client = None

        # --- Certification mode (§28/§29/§30) ---
        self.certification_mode = bool(getattr(self.settings, "certification_mode", False))
        # Cache is used only when explicitly enabled AND not certifying: a cached
        # result must never stand in as proof that the models ran this run (§30).
        self.cache_enabled = (
            bool(getattr(self.settings, "cache_enabled", True))
            and bool(getattr(self.settings, "verifier_cache_enabled", True))
            and not self.certification_mode
        )
        if self.certification_mode:
            self.logger.warning(
                "CERTIFICATION MODE ENABLED — fail-closed: detector/BGE/NLI fallback "
                "and mock/empty/malformed evidence will raise CertificationError; cache bypassed."
            )
        # Last-run model execution diagnostics (populated per sub-claim).
        self._last_reranker_diag: Dict[str, Any] | None = None
        self._last_nli_diag: Dict[str, Any] | None = None

    @property
    def n8n_client(self):
        if self._n8n_client is None:
            try:
                from services.n8n_retrieval_client import N8NRetrievalClient
            except ImportError:
                import sys
                from pathlib import Path
                root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                from services.n8n_retrieval_client import N8NRetrievalClient

            self._n8n_client = N8NRetrievalClient(
                webhook_url=self.settings.n8n_retrieval_webhook_url,
                health_url=self.settings.n8n_health_webhook_url,
                auth_mode=self.settings.n8n_auth_mode,
                header_name=getattr(self.settings, "n8n_header_name", "X-API-Key"),
                webhook_secret=self.settings.n8n_webhook_secret,
                timeout_seconds=self.settings.n8n_timeout_seconds,
            )
        return self._n8n_client

    # ------------------------------------------------------------------
    # Certification enforcement + model health (§28/§29)
    # ------------------------------------------------------------------
    def _enforce_certification_reranker(self, diag: Dict[str, Any] | None) -> None:
        from api.certification import enforce_reranker
        enforce_reranker(diag or {}, self.certification_mode)

    def _enforce_certification_nli(
        self, diag: Dict[str, Any] | None, evidence_present: bool
    ) -> None:
        from api.certification import enforce_nli
        enforce_nli(diag or {}, self.certification_mode, evidence_present)

    def _enforce_certification_evidence(self, raw_passages: List[Passage]) -> None:
        from api.certification import enforce_retrieval_evidence
        enforce_retrieval_evidence(
            raw_passages,
            self.certification_mode,
            bool(getattr(self.settings, "mock_mode", False)),
        )

    def health_check(self, probe: bool = True) -> Dict[str, Any]:
        """Explicit verifier model health (§29): BGE_READY / NLI_READY / N8N_READY.

        When ``probe`` is True this actually attempts to load BGE and NLI (real
        readiness, not inferred from imports). DETECTOR_READY is reported by the
        detector/service layer, not here.
        """
        bge_ready = False
        nli_ready = False
        if probe:
            try:
                self.reranker._load_model()
                bge_ready = self.reranker.model is not None and self.reranker._is_available
            except Exception as exc:
                self.logger.warning("BGE health probe failed: %s", exc)
            try:
                self.nli_engine._load_model()
                nli_ready = self.nli_engine.is_loaded()
            except Exception as exc:
                self.logger.warning("NLI health probe failed: %s", exc)
        n8n_ready = bool(
            getattr(self.settings, "n8n_retrieval_enabled", False)
            and getattr(self.settings, "n8n_retrieval_webhook_url", "")
            and (
                getattr(self.settings, "n8n_auth_mode", "none") == "none"
                or getattr(self.settings, "n8n_webhook_secret", None)
            )
        )
        return {
            "BGE_READY": bge_ready,
            "NLI_READY": nli_ready,
            "N8N_READY": n8n_ready,
            "CERTIFICATION_MODE": self.certification_mode,
            "CACHE_ENABLED": self.cache_enabled,
        }

    # ------------------------------------------------------------------
    # Evidence gating and confidence calibration
    # ------------------------------------------------------------------
    @staticmethod
    def _evidence_ranking_enabled() -> bool:
        import os

        raw = os.getenv("HALLUCIGUARD_EVIDENCE_RANKING")
        if raw is None or not raw.strip():
            return True
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _apply_evidence_ranking(
        cls,
        passages: List[Passage],
        *,
        claim_text: str,
        search_queries: List[str],
        domain: str = "general",
    ) -> List[Passage]:
        """Rank passages by deterministic relevance signals and keep only the
        decision-grade tier (STRONG + selected USABLE), before NLI.

        Surviving passages get ``relevance_score`` set to the computed relevance
        so the existing selection gate keeps them. Fail-open: on any error the
        original passages are returned unchanged so a ranking bug never blocks
        verification.
        """
        if not cls._evidence_ranking_enabled() or not passages:
            return passages
        try:
            from services.evidence_ranker import rank_and_select

            selected, _ranked = rank_and_select(
                passages, claim_text=claim_text, search_queries=search_queries, domain=domain
            )
            if not selected:
                # Every candidate was irrelevant. Returning [] routes the claim to
                # UNVERIFIED (safe) rather than letting a distractor reach NLI.
                return []
            return [
                item.passage.model_copy(update={"relevance_score": round(item.score, 4)})
                for item in selected
            ]
        except Exception:
            return passages

    @staticmethod
    def _select_relevant_passages(passages: List[Passage]) -> List[Passage]:
        """Apply relevance and source-diversity gates before NLI."""
        if not passages:
            return []

        # Filter out passages with low relevance (< 0.20) if any relevant passage exists
        relevant = [
            p for p in passages
            if float(getattr(p, "relevance_score", 0.0) or 0.0) >= 0.20
            or (float(getattr(p, "relevance_score", 0.0) or 0.0) == 0.0 and float(getattr(p, "source_confidence_hint", 0.0) or 0.0) >= 0.50)
        ]
        if not relevant:
            relevant = passages[:5]

        # Deduplicate multiple chunks from the same specific source URL/domain
        strongest_by_source: Dict[str, Passage] = {}
        for passage in relevant:
            source_key = (passage.url or passage.source_id or f"{passage.source}:{passage.title}").strip().lower()
            current = strongest_by_source.get(source_key)
            if current is None or (passage.relevance_score or 0.0) > (current.relevance_score or 0.0):
                strongest_by_source[source_key] = passage
        return list(strongest_by_source.values())

    @staticmethod
    def _subject_only_in_possessive_fragment(claim, passage):
        import re
        # A fragment about "Earth's Moon" does not assert that Earth itself
        # orbits anything. Reject this scope error rather than refute the claim.
        match = re.match(r"\s*(?:the\s+)?([A-Za-z][A-Za-z -]*?)\s+(?:orbits|is|was|are|were)\b", claim, re.I)
        if not match:
            return False
        subject = match.group(1).strip()
        text = passage.snippet
        possessive = r"\b" + re.escape(subject) + r"(?:&#0?39;|['’])s\s+[^.!?]+(?=[.!?]|$)"
        if not re.search(possessive, text, re.I):
            return False
        remaining = re.sub(possessive, "", text, flags=re.I)
        return not re.search(r"\b" + re.escape(subject) + r"\b", remaining, re.I)

    @staticmethod
    def _capital_premise(claim, passage, relation_verifier):
        """Normalize only whitespace segmentation of the same capital name,
        anchored to the same country. Keep original evidence/provenance intact.
        No aliases, translated names, or positive verdicts are synthesized.
        """
        import re
        claims = relation_verifier.extract_triples(claim) if relation_verifier else []
        facts = relation_verifier.extract_triples(passage.snippet) if relation_verifier else []
        def country(name):
            return re.sub(r"^(?:(?:socialist|democratic|federal) )?(?:republic|kingdom|state) of ", "", name)
        for c in claims:
            for f in facts:
                if (c.relation == f.relation == "capital_of" and not c.negated and not f.negated
                        and country(c.object) == country(f.object)
                        and c.subject != f.subject
                        and c.subject.replace(" ", "") == f.subject.replace(" ", "")):
                    normalized = re.sub(r"\b" + re.escape(f.subject) + r"\b",
                        c.subject, passage.snippet, flags=re.I)
                    return normalized, ["capital_name_whitespace_segmentation"]
        return passage.snippet, []

    @staticmethod
    def _historical_capital_only(claim, passage):
        import re
        # Explicitly historical assertions are not counter-evidence about the
        # current capital. A current assertion (including negation) stays in.
        if not re.search(r"\bis\b", claim, re.I) or re.search(r"\b(?:was|formerly|1\d{3}|20\d{2})\b", claim, re.I):
            return False
        sentences = [s for s in re.split(r"[.!?\n]", passage.snippet) if re.search(r"\bcapital\b", s, re.I)]
        return bool(sentences) and all(
            re.search(r"\b(?:ancient|former|historical|historic)\s+capital\b|\bwas\b[^.!?]*\bcapital\b", s, re.I)
            and not re.search(r"\b(?:currently|today|now|no longer)\b|\b(?:is|isn't|isn’t)\s+(?:not\s+)?(?:the\s+)?(?:current\s+|national\s+)?capital\b", s, re.I)
            for s in sentences)

    @staticmethod
    def _select_decision_grade_evidence(
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
        claim: str = "",
        relation_verifier: Any = None,
    ) -> Tuple[List[Passage], List[Dict[str, Any]]]:
        """Keep only evidence with a meaningful entailment/contradiction signal or structured relation match/mismatch."""
        pairs = list(zip(passages, nli_results))
        if not pairs:
            return [], []

        # Selection must not rewrite neural probabilities. Relation extraction is
        # diagnostic and may restrict off-topic evidence, never synthesize scores.
        selected = []
        aggregate = relation_verifier.verify_relation(claim, passages) if relation_verifier and claim else None
        subject = aggregate.claim_triple.subject if aggregate and aggregate.claim_triple else ""
        if aggregate and aggregate.claim_triple and aggregate.claim_triple.relation == "capital_of":
            # A wrong city's absence is expected in counter-evidence. Anchor the
            # topicality gate to the country, without changing NLI or its scores.
            subject = aggregate.claim_triple.object
        for passage, raw_result in pairs:
            if VerificationPipeline._subject_only_in_possessive_fragment(claim, passage):
                continue
            if (aggregate and aggregate.claim_triple and aggregate.claim_triple.relation == "capital_of"
                    and VerificationPipeline._historical_capital_only(claim, passage)):
                continue
            if raw_result.get("degraded") or raw_result.get("nli_degraded") or raw_result.get("validity_factor", 1.0) == 0:
                continue
            try:
                entailment = float(raw_result.get("entailment_score", 0.0))
                contradiction = float(raw_result.get("contradiction_score", 0.0))
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (entailment, contradiction)):
                    continue
            except (TypeError, ValueError):
                continue
            relation = relation_verifier.verify_relation(claim, [passage]) if relation_verifier and claim else None
            if (aggregate and aggregate.claim_triple and relation and contradiction > entailment
                    and aggregate.claim_triple.relation in {"created_by", "leads", "location_of"}
                    and not any(t.relation == aggregate.claim_triple.relation for t in relation.evidence_triples)):
                # Topical overlap is not evidence about the claimed relation.
                continue
            if subject and contradiction > entailment:
                context = relation_verifier._normalize_name(f"{passage.title} {passage.snippet}")
                if not set(subject.split()).issubset(set(context.split())):
                    continue
            result = dict(raw_result)
            result["relation_status"] = relation.status if relation else "NO_TRIPLE_EXTRACTED"
            result["score_provenance"] = "nli"
            if max(entailment, contradiction) >= 0.35:
                selected.append((passage, result))

        if not selected:
            return [], []

        selected.sort(
            key=lambda pair: max(
                float(pair[1].get("entailment_score", 0.0)),
                float(pair[1].get("contradiction_score", 0.0)),
            ),
            reverse=True,
        )
        selected = selected[:5]
        return [p for p, _ in selected], [r for _, r in selected]

    @staticmethod
    def _calibrate_confidence(
        scores: Dict[str, float],
        evidence_count: int,
        conflict_res: Dict[str, Any],
    ) -> float:
        """
        Calibrate confidence across both verified and contradicted claims.
        Reflects evidence strength, consensus, and source stability without arbitrary zeros.
        """
        if scores.get("confidence_score") is not None and float(scores.get("confidence_score", 0.0)) > 0.0:
            base = float(scores["confidence_score"])
        else:
            sup = float(scores.get("support_score", 0.0))
            con = float(scores.get("contradiction_score", 0.0))
            primary = max(sup, con)
            if primary < 0.15:
                return 0.0
            consensus = max(0.10, 1.0 - min(sup, con))
            base = primary * consensus

        if evidence_count <= 0 or base <= 0.0:
            return 0.0

        # EvidenceScorer already applies source-count calibration.  Applying a
        # second count penalty here made a single authoritative, 0.97-NLI
        # passage look like ~0.20 confidence.  Preserve the evidence-derived
        # score and only apply a genuine-conflict penalty at this layer.
        conflict_penalty = (
            0.60 if conflict_res.get("resolution_type") == "genuine_conflict" else 1.0
        )
        return round(max(0.0, min(1.0, base * conflict_penalty)), 4)

    # ------------------------------------------------------------------
    # Main pipeline entry
    # ------------------------------------------------------------------
    async def verify(self, payload: VerifierInputV2) -> VerifierOutputV2:
        tracker = PerformanceTracker()
        start_time = time.time()

        request_id = payload.query_id or str(uuid.uuid4())
        self.logger.info(
            "Processing verification request %s for domain %s",
            request_id,
            payload.domain,
        )

        # ── Stage 1: Domain Validation & Model Routing ──────────────
        with tracker.track(PipelineStage.DOMAIN_VALIDATION):
            first_claim_text = (
                payload.suspicious_claims[0].text if payload.suspicious_claims else ""
            )
            validated_domain, is_domain_correct = self.domain_validator.validate(
                claim_text=first_claim_text,
                detector_domain=payload.domain,
            )

        route = self.model_router.route(validated_domain, first_claim_text)
        validated_domain = route.domain
        registry = get_registry()
        adapter = registry.get_adapter(validated_domain)

        runtime_models = RuntimeModelInfo(
            embedding_model=route.embedding_model,
            reranker_model=route.reranker_model,
            nli_model=route.nli_model,
            cross_encoder=route.cross_encoder,
            classification_model=route.classification_model,
            retrieval_strategy=route.retrieval_strategy,
            device=route.device,
            claim_complexity=route.claim_complexity,
            latency_budget=route.latency_budget,
            routing_reason=route.routing_reason,
        )

        claim_reports: List[ClaimReport] = []
        total_retrieved = 0
        any_cache_hit = False
        adapter_failures: List[str] = []

        for claim in payload.suspicious_claims:
            try:
                cached_data = (
                    await self.cache.get(validated_domain, claim.text)
                    if self.cache_enabled
                    else None
                )
                if cached_data:
                    self.metrics.record_cache_hit()
                    self.logger.info("Cache hit for claim: %s", claim.claim_id)
                    report = (
                        ClaimReport(**cached_data)
                        if isinstance(cached_data, dict)
                        else cached_data
                    )
                    # Cached inference is historical proof, not execution by
                    # this request. Copy it before rewriting current telemetry.
                    import copy
                    trace = copy.deepcopy(report.retrieval_trace or {})
                    historical = {k: copy.deepcopy(v) for k, v in trace.items()
                        if k in {"backend_execution", "retrieval_degraded", "reranker_execution",
                                 "nli_execution", "evidence_flow", "subclaim_executions"}}
                    trace = {}
                    for component in ("reranker_execution", "nli_execution"):
                        trace[component] = {"status": "cache_hit", "attempted": False,
                            "inference_executed": False, "initialization_attempted": False,
                            "submitted_inputs": [], "degraded": False}
                    trace["backend_execution"] = {"route": "cache_hit", "sparse_attempted": False,
                        "dense_attempted": False, "sparse_executed": False, "dense_executed": False,
                        "fusion_executed": False, "selected_count": 0}
                    trace["evidence_flow"] = {"cache_reused_evidence_count": len(report.evidence),
                        "nli_selected": 0, "selected_passages": []}
                    trace["subclaim_executions"] = []
                    trace["execution_origin"] = "cache"
                    trace["cached_execution"] = historical
                    report = report.model_copy(update={"retrieval_trace": trace})
                    # The cache is keyed by (domain, claim TEXT); the stored report
                    # still carries the claim_id/claim_text of whatever run first
                    # populated it. Rebind them to the CURRENT request so the graph's
                    # exact claim-coverage contract (claim_id + text) holds on a
                    # warm/cross-pass cache hit and on same-text duplicate claims
                    # within one call (audit H4/M5/M11). Evidence is text-keyed, so
                    # only the request-scoped identifiers are corrected.
                    try:
                        report = report.model_copy(
                            update={"claim_id": claim.claim_id, "claim_text": claim.text}
                        )
                    except Exception:
                        try:
                            report.claim_id = claim.claim_id
                            report.claim_text = claim.text
                        except Exception:
                            pass
                    claim_reports.append(report)
                    any_cache_hit = True
                    continue
                else:
                    self.metrics.record_cache_miss()

                with tracker.track(PipelineStage.CLAIM_DECOMPOSITION):
                    # Feed RAW-CASE text to the decomposer. claim_normalizer lowercases,
                    # but the decomposer's subject/pronoun detection is case-sensitive
                    # (matches ^[A-Z]... and proper-noun POS), so passing normalized
                    # (lowercased) text silently broke subject extraction. The decomposer
                    # does its own internal cleaning.
                    sub_claims = self.claim_decomposer.decompose(claim.text)

                claim_evidence_items: List[EvidenceItem] = []
                sub_reports: List[Dict[str, Any]] = []
                claim_retrieved = 0
                claim_reranked = 0
                subclaim_executions = []
                backend_diag = {"route": "not_run", "degraded": True}

                for sub_claim in sub_claims:
                    with tracker.track(PipelineStage.QUERY_EXPANSION):
                        expanded_query, resolution = (
                            self.query_expander.resolve_and_expand(
                                sub_claim, validated_domain
                            )
                        )

                    with tracker.track(PipelineStage.RETRIEVAL):
                        retrieval_start = time.time()
                        all_raw: List[Passage] = []
                        n8n_trace_obj = None

                        # Object-anchored factual queries for this sub-claim. For
                        # relational claims ("X is the founder of Y") these anchor
                        # on the OBJECT entity ("Y founder", "who is the founder of
                        # Y", "Y founded by"); for a plain claim it is just [claim].
                        search_queries = self.query_expander.generate_search_queries(
                            sub_claim, validated_domain
                        )
                        if not search_queries:
                            search_queries = [expanded_query]
                        raw_lc = sub_claim.strip().lower()
                        # Queries that reframe the claim onto the checkable object
                        # entity — the ones that surface the ground truth a
                        # fabricated subject would otherwise hide from retrieval.
                        anchored_queries = [
                            q for q in search_queries if q.strip().lower() != raw_lc
                        ]

                        # ── N8N Retrieval Service V2 Integration ──────────────
                        if getattr(self.settings, "n8n_retrieval_enabled", True):
                            force_tav = (payload.retrieval_mode == "tavily_only")
                            n8n_res = await self.n8n_client.retrieve_evidence(
                                claim=sub_claim,
                                domain=validated_domain,
                                retrieval_mode=payload.retrieval_mode,
                                force_tavily=force_tav,
                                max_results=5,
                                request_id=request_id,
                                queries=search_queries,
                            )
                            n8n_trace_obj = n8n_res.trace
                            if n8n_res.success and n8n_res.passages:
                                all_raw.extend(n8n_res.passages)
                            elif not n8n_res.success:
                                self.logger.warning(
                                    "n8n retrieval failed (%s); falling back to Python adapters.",
                                    n8n_res.error,
                                )
                                adapter_failures.append(f"n8n:{str(n8n_res.error)[:80]}")

                        # ── Retrieval query set through Python adapters ───────
                        # If n8n produced nothing, run the FULL query set (classic
                        # fallback). If n8n DID return evidence, still issue the
                        # object-anchored queries and merge: retrieving on the raw
                        # claim alone biases toward the (possibly fabricated)
                        # subject, so without this the Corrector only ever sees a
                        # same-name distractor and can never recover the real fact.
                        # Non-relational claims add no extra queries here.
                        adapter_queries = anchored_queries if all_raw else search_queries
                        for q in adapter_queries:
                            try:
                                search_kwargs = {}
                                if hasattr(adapter, 'last_retrieval_trace'):
                                    search_kwargs['retrieval_mode'] = payload.retrieval_mode
                                if getattr(payload, 'source_mode', None):
                                    search_kwargs['source_mode'] = payload.source_mode
                                q_passages = await adapter.search(q, **search_kwargs)
                                all_raw.extend(q_passages)
                            except Exception as e:
                                self.logger.error(
                                    "Adapter retrieval failed (%s)",
                                    type(e).__name__,
                                )
                                adapter_failures.append(
                                    f"{validated_domain}:{type(e).__name__}"
                                )

                        # Attach n8n trace to adapter trace or initialize retrieval trace
                        if n8n_trace_obj:
                            if getattr(adapter, "last_retrieval_trace", None):
                                adapter.last_retrieval_trace.n8n_trace = n8n_trace_obj
                            else:
                                from schemas.retrieval_trace import RetrievalTrace
                                adapter.last_retrieval_trace = RetrievalTrace(
                                    requested_domain=validated_domain,
                                    primary_adapter=getattr(adapter, "name", "n8n_retrieval_service_v2"),
                                    retrieval_mode=payload.retrieval_mode,
                                    n8n_trace=n8n_trace_obj,
                                )

                        # Deduplicate raw passages by unique URL/ID/title
                        seen_doc_keys = set()
                        raw_passages = []
                        for p in all_raw:
                            doc_k = (p.url or p.source_id or p.title or "").strip().lower()
                            if doc_k and doc_k not in seen_doc_keys:
                                seen_doc_keys.add(doc_k)
                                raw_passages.append(p)

                        retrieval_duration = int((time.time() - retrieval_start) * 1000)
                        self.metrics.record_retrieval(
                            retrieval_duration, len(raw_passages)
                        )
                        total_retrieved += len(raw_passages)
                        claim_retrieved += len(raw_passages)

                        # §28 certification: refuse mock/empty/malformed evidence.
                        self._enforce_certification_evidence(raw_passages)

                        # Deterministic multi-signal relevance gate (§18/§19/§25):
                        # retrieval -> RELEVANCE -> (aggregation/rerank/NLI). Prune
                        # irrelevant evidence (e.g. a same-name distractor) BEFORE
                        # NLI so it can never drive a false contradiction. Runs
                        # here, in the Verifier, because relevance is a
                        # verification concern; the n8n client stays a faithful
                        # retrieval boundary. Fail-open + env-gated.
                        raw_passages = self._apply_evidence_ranking(
                            raw_passages,
                            claim_text=sub_claim,
                            search_queries=list(dict.fromkeys([expanded_query, *search_queries])),
                            domain=validated_domain,
                        )

                        from schemas.retrieval_trace import RetrievalTrace
                        if getattr(adapter, "last_retrieval_trace", None) is None:
                            adapter.last_retrieval_trace = RetrievalTrace(
                                requested_domain=validated_domain, primary_adapter=adapter.name,
                                retrieval_mode=payload.retrieval_mode,
                            )
                        adapter.last_retrieval_trace.evidence_flow = {
                            "raw_candidates": len(all_raw), "deduplicated_candidates": len(seen_doc_keys),
                            "after_relevance_ranking": len(raw_passages),
                        }


                    with tracker.track(PipelineStage.AGGREGATION):
                        # Relevance ranking is about TOPICALITY, not truth. For a
                        # relational claim the checkable proposition is
                        # object-anchored ("who is the founder of Y"), so rank
                        # candidates by that phrasing. Ranking by the subject-heavy
                        # raw claim ("Snehith is the founder of Microsoft") floats a
                        # same-name distractor to the top and prunes the passages
                        # that actually carry the true relation before NLI ever sees
                        # them. NLI + relation verification below still run against
                        # the RAW claim, so a contradiction is detected against the
                        # truth rather than hidden by it.
                        relevance_query = anchored_queries[0] if anchored_queries else sub_claim
                        aggregated_passages = self.aggregator.aggregate([raw_passages])
                        hybrid_passages = self.hybrid_retriever.retrieve(
                            relevance_query,
                            aggregated_passages,
                            k=8,
                            dense_model=route.dense_model,
                        )
                        backend_diag = (self.hybrid_retriever.diagnostics()
                                        if hasattr(self.hybrid_retriever, "diagnostics") else
                                        {"route": "untraced", "degraded": True})

                    with tracker.track(PipelineStage.RERANKING):
                        reranked_passages = self.reranker.rerank(
                            sub_claim,
                            hybrid_passages,
                            k=6,
                            model_name=route.reranker_model,
                        )
                        claim_reranked += len(reranked_passages)

                        # Update retrieval audit trace with real BGE score
                        if (reranked_passages and getattr(adapter, "last_retrieval_trace", None)
                                and self.reranker.diagnostics().get("inference_executed", False)):
                            trace_obj = getattr(adapter, "last_retrieval_trace")
                            top_bge = max([p.relevance_score for p in reranked_passages], default=0.0)
                            if trace_obj.gate_relevance_audit:
                                trace_obj.gate_relevance_audit.final_bge_relevance_score = round(top_bge, 4)
                                trace_obj.gate_relevance_audit.final_bge_score_available = True
                                trace_obj.gate_relevance_audit.signals_agree = (
                                    (trace_obj.gate_relevance_audit.gate_time_relevance_signal >= 0.25 and top_bge >= 0.25)
                                    or (trace_obj.gate_relevance_audit.gate_time_relevance_signal < 0.25 and top_bge < 0.25)
                                )

                        # §13/§26 BGE execution proof — record real-inference vs. fallback.
                        self._last_reranker_diag = self.reranker.diagnostics()
                        _rr_trace = getattr(adapter, "last_retrieval_trace", None)
                        if _rr_trace is not None:
                            from schemas.retrieval_trace import ModelExecutionTrace
                            _rr_trace.backend_execution = backend_diag
                            _rr_trace.retrieval_degraded = bool(
                                backend_diag.get("degraded", True) or self._last_reranker_diag.get("degraded", True)
                            )
                            _rr_trace.reranker_execution = ModelExecutionTrace(
                                **self._last_reranker_diag
                            )
                        # §28 certification: a fallback BGE score must never be certified.
                        self._enforce_certification_reranker(self._last_reranker_diag)

                    with tracker.track(PipelineStage.NLI):
                        nli_start = time.time()
                        relevant_passages = self._select_relevant_passages(
                            reranked_passages
                        )
                        nli_results = (
                            self.nli_engine.batch_classify(
                                sub_claim,
                                [self._capital_premise(sub_claim, p, self.evidence_scorer.relation_verifier)[0]
                                 for p in relevant_passages],
                                model_name=route.nli_model,
                            )
                        )
                        self.metrics.record_nli_inference(
                            int((time.time() - nli_start) * 1000)
                        )

                        # §15/§26 DeBERTa execution proof — record real-inference vs. fallback.
                        self._last_nli_diag = self.nli_engine.diagnostics()
                        _nli_trace = getattr(adapter, "last_retrieval_trace", None)
                        if _nli_trace is not None:
                            from schemas.retrieval_trace import ModelExecutionTrace
                            _nli_trace.nli_execution = ModelExecutionTrace(
                                **self._last_nli_diag
                            )
                        # §28 certification: degraded NLI on real evidence must not be certified.
                        self._enforce_certification_nli(
                            self._last_nli_diag, evidence_present=bool(relevant_passages)
                        )

                        # Run relation verification and attach trace
                        rel_check = self.evidence_scorer.relation_verifier.verify_relation(
                            sub_claim, relevant_passages
                        )
                        if getattr(adapter, "last_retrieval_trace", None):
                            trace_obj = getattr(adapter, "last_retrieval_trace")
                            from schemas.retrieval_trace import RelationCheckTrace
                            c_t = rel_check.claim_triple.model_dump() if rel_check.claim_triple else {}
                            e_ts = [t.model_dump() for t in rel_check.evidence_triples]
                            trace_obj.relation_check = RelationCheckTrace(
                                claim_subject=c_t.get("subject", ""),
                                claim_relation=c_t.get("relation", ""),
                                claim_object=c_t.get("object", ""),
                                evidence_subject=e_ts[0].get("subject", "") if e_ts else "",
                                evidence_relation=e_ts[0].get("relation", "") if e_ts else "",
                                evidence_object=e_ts[0].get("object", "") if e_ts else "",
                                check_result=rel_check.status,
                                combination_rule_applied=rel_check.combination_rule_applied or "STANDARD_NLI_SCORING",
                            )

                    # Only evidence with a meaningful NLI or relation signal is allowed to
                    # influence scoring, conflict resolution, or final citations.
                    decision_passages, decision_nli = (
                        self._select_decision_grade_evidence(
                            relevant_passages,
                            nli_results,
                            claim=sub_claim,
                            relation_verifier=self.evidence_scorer.relation_verifier,
                        )
                    )
                    current_trace = getattr(adapter, "last_retrieval_trace", None)
                    if current_trace is not None:
                        current_trace.evidence_flow.update({
                            "hybrid_selected": len(hybrid_passages), "reranker_selected": len(reranked_passages),
                            "nli_selected": len(relevant_passages), "decision_grade_selected": len(decision_passages),
                            "selected_passages": [{
                                "source_id": p.source_id, "source": p.source,
                                "snippet_sha256": hashlib.sha256(p.snippet.encode()).hexdigest(),
                                "characters": len(p.snippet),
                                "model_input_sha256": hashlib.sha256(self._capital_premise(
                                    sub_claim, p, self.evidence_scorer.relation_verifier)[0].encode()).hexdigest(),
                                "input_transforms": self._capital_premise(
                                    sub_claim, p, self.evidence_scorer.relation_verifier)[1],
                            } for p in relevant_passages],
                            "nli_outputs": [{
                                "source_id": p.source_id,
                                "label": str(getattr(n.get("label"), "value", n.get("label"))),
                                "entailment": n.get("entailment_score"),
                                "contradiction": n.get("contradiction_score"),
                                "neutral": n.get("neutral_score"),
                                "degraded": bool(n.get("degraded", False)),
                                "selected_for_decision": p in decision_passages,
                                "rejection_reason": (None if p in decision_passages else
                                    "subject_only_in_possessive_fragment" if self._subject_only_in_possessive_fragment(sub_claim, p)
                                    else "historical_scope_mismatch" if self._historical_capital_only(sub_claim, p)
                                    else "no_decision_grade_signal_or_relation_gate"),
                            } for p, n in zip(relevant_passages, nli_results)],
                        })
                        subclaim_executions.append({
                            "subclaim_sha256": hashlib.sha256(sub_claim.encode()).hexdigest(),
                            "backend_execution": backend_diag,
                            "reranker_execution": self._last_reranker_diag,
                            "nli_execution": self._last_nli_diag,
                            "evidence_flow": dict(current_trace.evidence_flow),
                        })
                        current_trace.subclaim_executions = list(subclaim_executions)

                    with tracker.track(PipelineStage.SCORING):
                        scores_dict = self.evidence_scorer.score_evidence(
                            claim=sub_claim,
                            passages=decision_passages,
                            nli_results=decision_nli,
                            domain=validated_domain,
                        )
                        formatted_evidence = self.citation_formatter.format_all(
                            passages=decision_passages,
                            nli_results=decision_nli,
                            domain=validated_domain,
                            reliability_manager=self.reliability_manager,
                            claim=sub_claim,
                        )
                        claim_evidence_items.extend(formatted_evidence)
                        sub_reports.append(
                            {
                                "scores": scores_dict,
                                "evidence_items": formatted_evidence,
                            }
                        )

                merged_sub_result = self.claim_merger.merge_results(sub_reports)
                overall_scores = merged_sub_result.get(
                    "scores",
                    {
                        "support_score": 0.0,
                        "contradiction_score": 0.0,
                        "trust_score": 0.0,
                    },
                )

                conflict_res = self.conflict_resolver.resolve(claim_evidence_items)

                verdict = merged_sub_result.get("verdict")
                if isinstance(verdict, str):
                    verdict_map = {
                        "verified": VerdictLabel.VERIFIED,
                        "contradicted": VerdictLabel.CONTRADICTED,
                        "unverified": VerdictLabel.UNVERIFIED,
                        "conflicted": VerdictLabel.CONFLICTED,
                    }
                    verdict = verdict_map.get(verdict, VerdictLabel.UNVERIFIED)
                elif not isinstance(verdict, VerdictLabel):
                    verdict = (
                        VerdictLabel.UNVERIFIED
                        if not claim_evidence_items
                        else VerdictLabel.CONFLICTED
                    )

                confidence = self._calibrate_confidence(
                    overall_scores, len(claim_evidence_items), conflict_res
                )

                supporting_sources = list(
                    {
                        e.source
                        for e in claim_evidence_items
                        if e.entailment_label == EntailmentLabel.ENTAILMENT
                    }
                )
                contradicting_sources = list(
                    {
                        e.source
                        for e in claim_evidence_items
                        if e.entailment_label == EntailmentLabel.CONTRADICTION
                    }
                )

                with tracker.track(PipelineStage.FORMATTING):
                    explanation = self.explanation_generator.generate(
                        claim_text=claim.text,
                        evidence_items=claim_evidence_items,
                        verdict=verdict,
                        scores=overall_scores,
                        conflict_resolution=conflict_res,
                    )

                    report = ClaimReport(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        verdict=verdict,
                        support_score=float(overall_scores.get("support_score", 0.0)),
                        contradiction_score=float(
                            overall_scores.get("contradiction_score", 0.0)
                        ),
                        trust_score=float(overall_scores.get("trust_score", 0.0)),
                        confidence_score=confidence,
                        evidence=claim_evidence_items,
                        explanation=explanation,
                        supporting_sources=supporting_sources,
                        contradicting_sources=contradicting_sources,
                        retrieved_documents=claim_retrieved,
                        reranked_documents=claim_reranked,
                        verified_evidence=len(claim_evidence_items),
                        retrieval_trace=(
                            getattr(adapter, "last_retrieval_trace", None).model_dump()
                            if getattr(adapter, "last_retrieval_trace", None) else
                            {"backend_execution": backend_diag,
                             "retrieval_degraded": bool(backend_diag.get("degraded", True) or
                                                        self._last_reranker_diag.get("degraded", True))}
                        ),
                    )

                if claim_evidence_items and self.cache_enabled:
                    await self.cache.set(
                        validated_domain, claim.text, report.model_dump()
                    )
                claim_reports.append(report)

            except CertificationError:
                # Fail-closed: a certification failure must surface as a controlled
                # error, never be masked as a benign UNVERIFIED verdict (§28/§45).
                self.logger.error(
                    "Certification failure while processing claim %s — aborting verification.",
                    claim.claim_id,
                )
                raise
            except Exception as e:
                self.logger.error(
                    "Pipeline error processing claim %s: %s",
                    claim.claim_id,
                    type(e).__name__,
                )
                error_report = ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    evidence=[],
                    support_score=0.0,
                    contradiction_score=0.0,
                    trust_score=0.0,
                    confidence_score=0.0,
                    verdict=VerdictLabel.UNVERIFIED,
                    explanation=f"Pipeline error: {type(e).__name__}. This claim could not be verified.",
                )
                claim_reports.append(error_report)

        latency_ms = int((time.time() - start_time) * 1000)
        self.metrics.record_request(validated_domain, latency_ms, success=True)

        overall_confidence = (
            # Full-response acceptance requires every atomic claim to clear the
            # evidence gate, so the least-certain claim is the safe aggregate.
            min(cr.confidence_score for cr in claim_reports)
            if claim_reports
            else 0.0
        )

        pipeline_stages_list = tracker.to_pipeline_stages()
        append_retrieval_health_stage(
            pipeline_stages_list,
            adapter_failures,
            total_retrieved,
        )

        sources_attempted = list(getattr(adapter, "sources_attempted", []))
        if not sources_attempted:
            collected = []
            for r in claim_reports:
                if r.retrieval_trace and hasattr(r.retrieval_trace, "sources_used"):
                    collected.extend(r.retrieval_trace.sources_used or [])
                elif r.retrieval_trace and isinstance(r.retrieval_trace, dict):
                    collected.extend(r.retrieval_trace.get("sources_used", []))
            sources_attempted = list(dict.fromkeys(collected)) or [adapter.name]
        sources_succeeded = getattr(adapter, "sources_succeeded", [adapter.name] if total_retrieved > 0 else [])
        sources_failed = getattr(adapter, "sources_failed", [] if total_retrieved > 0 else [adapter.name])

        final_response = VerifierOutputV2(
            query_id=payload.query_id,
            domain=validated_domain,
            domain_validated=is_domain_correct,
            adapter=adapter.name,
            sources_attempted=sources_attempted,
            sources_succeeded=sources_succeeded,
            sources_failed=sources_failed,
            retrieved_sources=total_retrieved,
            verified_sources=sum(len(r.evidence) for r in claim_reports),
            claim_evidence=claim_reports,
            overall_evidence_confidence=round(overall_confidence, 4),
            latency_ms=latency_ms,
            pipeline_stages=pipeline_stages_list,
            runtime_models=runtime_models,
            cache_hit=any_cache_hit,
        )

        return final_response
