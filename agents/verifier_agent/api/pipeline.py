from __future__ import annotations
import uuid
import time
from typing import List, Dict, Any, Optional

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
from models import get_model_manager
from utils.logging import setup_logger

class VerificationPipeline:
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
        tracker.start()
        
        request_id = str(uuid.uuid4())
        self.logger.info(f"Processing request {request_id}")
        
        # Stage 1: Domain Validation
        validated_domain = await self.domain_validator.validate(
            claim_text=payload.context if hasattr(payload, 'context') else "",
            domain=payload.domain
        )
        registry = get_registry()
        adapter = registry.get_adapter(validated_domain)
        
        claim_reports: List[ClaimReport] = []
        
        # Stage 2: Process each suspicious claim
        for claim in payload.suspicious_claims:
            try:
                # a. Cache Check
                cached_report = await self.cache.get(claim.claim_text)
                if cached_report:
                    claim_reports.append(cached_report)
                    continue

                # b. Decomposition
                normalized_claim = self.claim_normalizer.normalize(claim.claim_text)
                sub_claims = self.claim_decomposer.decompose(normalized_claim)
                
                all_evidence: List[EvidenceItem] = []
                
                for sub_claim in sub_claims:
                    # c. Query Expansion
                    expanded_queries = self.query_expander.expand(sub_claim, validated_domain)
                    
                    # d. Multi-source Retrieval
                    raw_passages: List[Passage] = []
                    for q in expanded_queries:
                        try:
                            results = await adapter.search(q)
                            raw_passages.extend(results)
                        except Exception as e:
                            self.logger.error(f"Retrieval failed for query {q}: {e}")
                    
                    # e. Aggregation + Dedup
                    aggregated = self.aggregator.aggregate(raw_passages)
                    
                    # f. Hybrid Retrieval
                    hybrid_results = self.hybrid_retriever.retrieve(sub_claim, aggregated)
                    
                    # g. Cross-encoder Reranking
                    reranked = await self.reranker.rerank(sub_claim, hybrid_results)
                    
                    # h. NLI Entailment
                    try:
                        for passage in reranked:
                            passage.entailment = await self.nli_engine.predict(sub_claim, passage.text)
                    except Exception as e:
                        self.logger.error(f"NLI model failed: {e}")
                        # Fallback to neutral if NLI fails
                        for passage in reranked:
                            passage.entailment = EntailmentLabel.NEUTRAL
                    
                    # i. Evidence Scoring
                    scored_evidence = self.evidence_scorer.score(reranked)
                    weighted_evidence = self.reliability_manager.apply_weights(scored_evidence, validated_domain)
                    all_evidence.extend(weighted_evidence)
                    
                # j. Conflict Resolution
                resolved_evidence, verdict = self.conflict_resolver.resolve(all_evidence)
                
                # k. Explanation Generation
                explanation = self.explanation_generator.generate(claim.claim_text, resolved_evidence, verdict)
                
                # l. Citation Formatting
                cited_evidence = self.citation_formatter.format(resolved_evidence)
                
                report = ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    verdict=verdict,
                    explanation=explanation,
                    evidence=cited_evidence
                )
                
                # m. Cache
                await self.cache.set(claim.claim_text, report)
                claim_reports.append(report)
                
            except Exception as e:
                self.logger.error(f"Failed to process claim {claim.claim_id}: {e}")
                claim_reports.append(ClaimReport(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    verdict=VerdictLabel.UNVERIFIABLE,
                    explanation=f"Internal error processing claim: {str(e)}",
                    evidence=[]
                ))
        
        # Stage 3: Assemble Response
        tracker.stop()
        final_output = self.response_formatter.format(
            request_id=request_id,
            domain=validated_domain,
            reports=claim_reports,
            metrics=tracker.get_metrics()
        )
        
        return final_output
