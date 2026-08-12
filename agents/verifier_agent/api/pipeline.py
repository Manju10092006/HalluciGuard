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
        self.citation_formatter = CitationFormatter()
        self.response_formatter = ResponseFormatter()
        self.domain_validator = DomainValidator()
        self.model_router = ModelRouter()
        self.query_expander = QueryExpander()
        self.cache = SqliteCache()
        self.metrics = MetricsCollector()
        self.logger = setup_logger("pipeline")

    # ------------------------------------------------------------------
    # Evidence gating and confidence calibration
    # ------------------------------------------------------------------
    @staticmethod
    def _select_decision_grade_evidence(
        passages: List[Passage],
        nli_results: List[Dict[str, Any]],
    ) -> Tuple[List[Passage], List[Dict[str, Any]]]:
        """Keep only evidence with a meaningful entailment/contradiction signal."""
        pairs = list(zip(passages, nli_results))
        if not pairs:
            return [], []

        selected = [
            (passage, result)
            for passage, result in pairs
            if not result.get("degraded", False)
            and max(
                float(result.get("entailment_score", 0.0)),
                float(result.get("contradiction_score", 0.0)),
            )
            >= 0.35
        ]

        # If the model is uncertain about every passage, do not manufacture
        # evidence. Return empty evidence and let the final verdict be
        # insufficient_evidence.
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
        """Calibrate trust without unfairly penalizing a single strong source."""
        base = max(0.0, min(1.0, float(scores.get("trust_score", 0.0))))
        if evidence_count <= 0 or base <= 0.0:
            return 0.0

        # Quality is already encoded in trust_score. Evidence count is only a
        # modest stability bonus, so one authoritative source can still yield
        # high confidence while independent sources increase stability.
        evidence_factor = 0.75 + 0.25 * min(1.0, evidence_count / 3.0)
        conflict_penalty = (
            0.80 if conflict_res.get("resolution_type") == "genuine_conflict" else 1.0
        )
        return round(max(0.0, min(1.0, base * evidence_factor * conflict_penalty)), 4)

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
                cached_data = await self.cache.get(validated_domain, claim.text)
                if cached_data:
                    self.metrics.record_cache_hit()
                    self.logger.info("Cache hit for claim: %s", claim.claim_id)
                    report = (
                        ClaimReport(**cached_data)
                        if isinstance(cached_data, dict)
                        else cached_data
                    )
                    claim_reports.append(report)
                    any_cache_hit = True
                    continue
                else:
                    self.metrics.record_cache_miss()

                with tracker.track(PipelineStage.CLAIM_DECOMPOSITION):
                    normalized_text = self.claim_normalizer.normalize(claim.text)
                    sub_claims = self.claim_decomposer.decompose(
                        normalized_text or claim.text
                    )

                claim_evidence_items: List[EvidenceItem] = []
                sub_reports: List[Dict[str, Any]] = []
                claim_retrieved = 0
                claim_reranked = 0

                for sub_claim in sub_claims:
                    with tracker.track(PipelineStage.QUERY_EXPANSION):
                        expanded_query, resolution = (
                            self.query_expander.resolve_and_expand(
                                sub_claim, validated_domain
                            )
                        )

                    with tracker.track(PipelineStage.RETRIEVAL):
                        retrieval_start = time.time()
                        try:
                            raw_passages = await adapter.search(expanded_query)
                        except Exception as e:
                            self.logger.error(
                                "Adapter retrieval failed for query '%s': %s",
                                expanded_query,
                                e,
                            )
                            adapter_failures.append(
                                f"{validated_domain}:{str(e)[:100]}"
                            )
                            raw_passages = []

                        retrieval_duration = int((time.time() - retrieval_start) * 1000)
                        self.metrics.record_retrieval(
                            retrieval_duration, len(raw_passages)
                        )
                        total_retrieved += len(raw_passages)
                        claim_retrieved += len(raw_passages)

                    with tracker.track(PipelineStage.AGGREGATION):
                        aggregated_passages = self.aggregator.aggregate([raw_passages])
                        hybrid_passages = self.hybrid_retriever.retrieve(
                            sub_claim,
                            aggregated_passages,
                            k=5,
                            dense_model=route.dense_model,
                        )

                    with tracker.track(PipelineStage.RERANKING):
                        reranked_passages = self.reranker.rerank(
                            sub_claim,
                            hybrid_passages,
                            k=5,
                            model_name=route.reranker_model,
                        )
                        claim_reranked += len(reranked_passages)

                    with tracker.track(PipelineStage.NLI):
                        nli_start = time.time()
                        nli_results = (
                            self.nli_engine.batch_classify(
                                sub_claim,
                                [p.snippet for p in reranked_passages],
                                model_name=route.nli_model,
                            )
                            if reranked_passages
                            else []
                        )
                        self.metrics.record_nli_inference(
                            int((time.time() - nli_start) * 1000)
                        )

                    # Only evidence with a meaningful NLI signal is allowed to
                    # influence scoring, conflict resolution, or final citations.
                    decision_passages, decision_nli = (
                        self._select_decision_grade_evidence(
                            reranked_passages, nli_results
                        )
                    )

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
                        "likely_hallucinated": VerdictLabel.LIKELY_HALLUCINATED,
                        "insufficient_evidence": VerdictLabel.INSUFFICIENT_EVIDENCE,
                    }
                    verdict = verdict_map.get(verdict, VerdictLabel.MIXED_EVIDENCE)
                elif not isinstance(verdict, VerdictLabel):
                    verdict = (
                        VerdictLabel.INSUFFICIENT_EVIDENCE
                        if not claim_evidence_items
                        else VerdictLabel.MIXED_EVIDENCE
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
                    )

                if claim_evidence_items:
                    await self.cache.set(
                        validated_domain, claim.text, report.model_dump()
                    )
                claim_reports.append(report)

            except Exception as e:
                self.logger.error(
                    "Pipeline error processing claim %s: %s",
                    claim.claim_id,
                    e,
                    exc_info=True,
                )
                error_report = ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    evidence=[],
                    support_score=0.0,
                    contradiction_score=0.0,
                    trust_score=0.0,
                    confidence_score=0.0,
                    verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                    explanation=f"Pipeline error: {type(e).__name__} — {str(e)[:200]}. This claim could not be verified.",
                )
                claim_reports.append(error_report)

        latency_ms = int((time.time() - start_time) * 1000)
        self.metrics.record_request(validated_domain, latency_ms, success=True)

        avg_trust = (
            sum(cr.trust_score for cr in claim_reports) / len(claim_reports)
            if claim_reports
            else 0.0
        )

        pipeline_stages_list = tracker.to_pipeline_stages()
        if adapter_failures:
            pipeline_stages_list.append(
                PipelineStageStatus(
                    stage=PipelineStage.RETRIEVAL,
                    status="failed",
                    duration_ms=0,
                    details=f"Adapter failures: {', '.join(adapter_failures)}",
                )
            )
        elif total_retrieved == 0:
            pipeline_stages_list.append(
                PipelineStageStatus(
                    stage=PipelineStage.RETRIEVAL,
                    status="degraded",
                    duration_ms=0,
                    details="No passages were retrieved; downstream reranking, NLI, and evidence scoring had no real evidence to process.",
                )
            )

        final_response = VerifierOutputV2(
            query_id=payload.query_id,
            domain=validated_domain,
            domain_validated=is_domain_correct,
            retrieved_sources=total_retrieved,
            verified_sources=sum(len(r.evidence) for r in claim_reports),
            claim_evidence=claim_reports,
            overall_evidence_confidence=round(avg_trust, 4),
            latency_ms=latency_ms,
            pipeline_stages=pipeline_stages_list,
            runtime_models=runtime_models,
            cache_hit=any_cache_hit,
        )

        return final_response
