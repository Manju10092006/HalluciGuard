from __future__ import annotations
import uuid
import time
import logging
from typing import List, Dict, Any

from schemas.models import (
    VerifierInputV2, VerifierOutputV2, ClaimReport, EvidenceItem,
    SuspiciousClaim, Passage, EntailmentLabel, VerdictLabel,
    PipelineStage, PipelineStageStatus
)
from adapters.registry import get_registry
from claims import ClaimDecomposer, ClaimNormalizer, ClaimMerger
from retrievers import HybridRetriever
from aggregation import EvidenceAggregator
from rerankers import CrossEncoderReranker
from nli import NLIEngine
from scorers import EvidenceScorer, SourceReliabilityManager, ConflictResolver
from explanations import ExplanationGenerator
from formatters import CitationFormatter, ResponseFormatter
from routers import DomainValidator, QueryExpander
from cache import SqliteCache
from metrics import MetricsCollector, PerformanceTracker
from utils.logging import setup_logger

class VerificationPipeline:
    """The 8-stage verification pipeline orchestrator."""

    def __init__(self) -> None:
        self.claim_decomposer = ClaimDecomposer()
        self.claim_normalizer = ClaimNormalizer()
        self.claim_merger = ClaimMerger()
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
        self.query_expander = QueryExpander()
        self.cache = SqliteCache()
        self.metrics = MetricsCollector()
        self.logger = setup_logger('pipeline')

    async def verify(self, payload: VerifierInputV2) -> VerifierOutputV2:
        tracker = PerformanceTracker()
        start_time = time.time()
        
        request_id = payload.query_id or str(uuid.uuid4())
        self.logger.info(f"Processing verification request {request_id} for domain {payload.domain}")

        # Stage 1: Domain Validation
        with tracker.track(PipelineStage.DOMAIN_VALIDATION):
            first_claim_text = payload.suspicious_claims[0].text if payload.suspicious_claims else ""
            validated_domain, is_domain_correct = self.domain_validator.validate(
                claim_text=first_claim_text,
                detector_domain=payload.domain
            )
        
        registry = get_registry()
        adapter = registry.get_adapter(validated_domain)
        
        claim_reports: List[ClaimReport] = []
        total_retrieved = 0

        # Stage 2: Process Each Suspicious Claim
        for claim in payload.suspicious_claims:
            claim_start_time = time.time()
            try:
                # Cache Lookup
                cached_data = await self.cache.get(validated_domain, claim.text)
                if cached_data:
                    self.metrics.record_cache_hit()
                    self.logger.info(f"Cache hit for claim: {claim.claim_id}")
                    report = ClaimReport(**cached_data) if isinstance(cached_data, dict) else cached_data
                    claim_reports.append(report)
                    continue
                else:
                    self.metrics.record_cache_miss()

                # Stage 2: Claim Decomposition
                with tracker.track(PipelineStage.CLAIM_DECOMPOSITION):
                    normalized_text = self.claim_normalizer.normalize(claim.text)
                    sub_claims = self.claim_decomposer.decompose(normalized_text or claim.text)

                claim_evidence_items: List[EvidenceItem] = []
                sub_reports: List[Dict[str, Any]] = []

                for sub_claim in sub_claims:
                    # Stage 3: Query Expansion
                    with tracker.track(PipelineStage.QUERY_EXPANSION):
                        expanded_query = self.query_expander.expand(sub_claim, validated_domain)

                    # Stage 4: Multi-Source Retrieval
                    with tracker.track(PipelineStage.RETRIEVAL):
                        retrieval_start = time.time()
                        try:
                            raw_passages = await adapter.search(expanded_query)
                        except Exception as e:
                            self.logger.error(f"Adapter retrieval failed for query '{expanded_query}': {e}")
                            raw_passages = []
                        
                        retrieval_duration = int((time.time() - retrieval_start) * 1000)
                        self.metrics.record_retrieval(retrieval_duration, len(raw_passages))
                        total_retrieved += len(raw_passages)

                    # Stage 5: Aggregation + Deduplication & Hybrid RRF Retrieval
                    with tracker.track(PipelineStage.AGGREGATION):
                        aggregated_passages = self.aggregator.aggregate([raw_passages])
                        hybrid_passages = self.hybrid_retriever.retrieve(sub_claim, aggregated_passages, k=5)

                    # Stage 6: Cross-Encoder Reranking
                    with tracker.track(PipelineStage.RERANKING):
                        reranked_passages = self.reranker.rerank(sub_claim, hybrid_passages, k=5)

                    # Stage 7: NLI Entailment
                    with tracker.track(PipelineStage.NLI):
                        nli_start = time.time()
                        nli_results = [self.nli_engine.classify(sub_claim, p.snippet) for p in reranked_passages]
                        self.metrics.record_nli_inference(int((time.time() - nli_start) * 1000))

                    # Stage 8: Evidence Scoring & Citation Formatting
                    with tracker.track(PipelineStage.SCORING):
                        scores_dict = self.evidence_scorer.score_evidence(
                            claim=sub_claim,
                            passages=reranked_passages,
                            nli_results=nli_results,
                            domain=validated_domain
                        )
                        formatted_evidence = self.citation_formatter.format_all(
                            passages=reranked_passages,
                            nli_results=nli_results,
                            domain=validated_domain,
                            reliability_manager=self.reliability_manager
                        )
                        claim_evidence_items.extend(formatted_evidence)
                        sub_reports.append({
                            'scores': scores_dict,
                            'evidence_items': formatted_evidence
                        })

                # Merge sub-claim results if multiple sub-claims
                merged_sub_result = self.claim_merger.merge_results(sub_reports)
                overall_scores = merged_sub_result.get('scores', {
                    'support_score': 0.0, 'contradiction_score': 0.0, 'trust_score': 0.0
                })

                # Stage 8: Conflict Resolution
                conflict_res = self.conflict_resolver.resolve(claim_evidence_items)

                # Verdict assignment
                verdict = merged_sub_result.get('verdict')
                if isinstance(verdict, str):
                    if verdict == 'verified':
                        verdict = VerdictLabel.VERIFIED
                    elif verdict == 'likely_hallucinated':
                        verdict = VerdictLabel.LIKELY_HALLUCINATED
                    else:
                        verdict = VerdictLabel.MIXED_EVIDENCE
                elif not isinstance(verdict, VerdictLabel):
                    verdict = VerdictLabel.INSUFFICIENT_EVIDENCE if not claim_evidence_items else VerdictLabel.MIXED_EVIDENCE

                # Stage 8: Explanation Generation & Citation Formatting
                with tracker.track(PipelineStage.FORMATTING):
                    explanation = self.explanation_generator.generate(
                        claim_text=claim.text,
                        evidence_items=claim_evidence_items,
                        verdict=verdict,
                        scores=overall_scores,
                        conflict_resolution=conflict_res
                    )

                    report = self.response_formatter.format_claim_report(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        evidence_items=claim_evidence_items,
                        scores=overall_scores,
                        explanation=explanation,
                        verdict=verdict
                    )

                # Cache set (only cache valid non-empty evidence results)
                if claim_evidence_items:
                    await self.cache.set(validated_domain, claim.text, report.model_dump())
                claim_reports.append(report)

            except Exception as e:
                self.logger.error(f"Pipeline error processing claim {claim.claim_id}: {e}", exc_info=True)
                error_report = ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    evidence=[],
                    support_score=0.0,
                    contradiction_score=0.0,
                    trust_score=0.0,
                    verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                    explanation=f"Processing encountered error: {str(e)}"
                )
                claim_reports.append(error_report)

        latency_ms = int((time.time() - start_time) * 1000)
        self.metrics.record_request(validated_domain, latency_ms, success=True)

        final_response = self.response_formatter.format_full_response(
            query_id=payload.query_id,
            domain=validated_domain,
            domain_validated=is_domain_correct,
            claim_reports=claim_reports,
            latency_ms=latency_ms,
            pipeline_stages=tracker.to_pipeline_stages(),
            retrieved_sources=total_retrieved,
            verified_sources=sum(len(r.evidence) for r in claim_reports)
        )

        return final_response
