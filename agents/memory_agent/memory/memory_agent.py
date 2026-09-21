from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from ..cache.verification_cache import VerificationCache
from ..config.settings import Settings, get_settings
from ..contradiction.detector import ContradictionDetector
from ..knowledge_graph.graph import KnowledgeGraph
from ..patterns.pattern_learner import PatternLearner
from ..storage.journal import StorageJournal
from ..schemas.models import (
    BatchStoreResponse,
    CacheStats,
    CachedVerification,
    ContradictionAlert,
    DeleteFactResponse,
    EntityType,
    HallucinationPattern,
    MemoryStatsResponse,
    PatternType,
    RecallRequest,
    RecallResponse,
    RelationType,
    SourceTrustRecord,
    StoreFactRequest,
    StoreFactResponse,
    TrustChangeReason,
    TrustUpdate,
    UpdateFactRequest,
    UpdateFactResponse,
    VectorSearchResult,
)
from ..trust.source_trust import SourceTrustManager
from ..vector_store.faiss_store import VectorStore

logger = logging.getLogger(__name__)

_BLOCKING_VERDICTS = (
    "contradicted", "likely_hallucinated", "unverified", "conflicted",
)

_BLOCKING_STATUSES = {
    "CONTRADICTED",
    "CONTRADICTION",
    "NOT_ENOUGH_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    "UNCERTAIN",
    "UNVERIFIED",
    "REJECTED",
    "ABSTAINED",
    "ABSTAIN",
}


class MemoryAgent:
    """Orchestrator that ties all memory subsystems together."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        cache: Optional[VerificationCache] = None,
        pattern_learner: Optional[PatternLearner] = None,
        source_trust: Optional[SourceTrustManager] = None,
        vector_store: Optional[VectorStore] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        journal: Optional[StorageJournal] = None,
    ):
        self._settings = settings or get_settings()
        self.kg = knowledge_graph or KnowledgeGraph(
            persistence_path=self._settings.kg_persistence_path,
            max_nodes=self._settings.kg_max_nodes,
            weight_decay=self._settings.kg_edge_weight_decay,
        )
        self.cache = cache or VerificationCache(
            db_path=self._settings.cache_db_path,
            ttl=self._settings.cache_ttl,
        )
        self.patterns = pattern_learner or PatternLearner(
            db_path=self._settings.pattern_db_path,
            min_support=self._settings.pattern_min_support,
            confidence_threshold=self._settings.pattern_confidence_threshold,
        )
        self.trust = source_trust or SourceTrustManager(
            db_path=self._settings.trust_db_path,
            prior=self._settings.trust_prior,
            learning_rate=self._settings.trust_learning_rate,
            decay_rate=self._settings.trust_decay_rate,
        )
        self.vectors = vector_store or VectorStore(
            embedding_model=self._settings.embedding_model,
            store_path=self._settings.vector_store_path,
            dimension=self._settings.vector_dimension,
            top_k=self._settings.vector_top_k,
        )
        self.contradictions = contradiction_detector or ContradictionDetector(
            nli_model=self._settings.nli_contradiction_model,
            use_nli=self._settings.nli_contradiction_check,
            nli_threshold=self._settings.nli_contradiction_threshold,
        )
        self.journal = journal or StorageJournal(
            db_path=self._settings.storage_journal_path,
        )

    async def initialize(self) -> None:
        await self.cache.initialize()
        await self.patterns.initialize()
        await self.trust.initialize()
        await self.journal.initialize()
        self.vectors.initialize()
        logger.info("Memory agent initialized")

    async def close(self) -> None:
        """Best-effort shutdown: persist and release every subsystem.

        Memory is the terminal, audit-only stage of the pipeline. A save or
        close failure in one subsystem must NOT (a) skip shutdown of the others
        (leaking DB connections/file handles) nor (b) raise out of the caller's
        ``finally: await close()`` and turn a Judge-accepted answer into a memory
        node failure. Every step is attempted; failures are logged and swallowed.
        """
        # Ordered (persist first, then release), each guarded independently.
        steps: list[tuple[str, Callable[..., Any]]] = [
            ("knowledge_graph.save", self.kg.save),
            ("vector_store.save", self.vectors.save),
            ("cache.close", self.cache.close),
            ("patterns.close", self.patterns.close),
            ("trust.close", self.trust.close),
        ]
        journal = getattr(self, "journal", None)
        if journal is not None:
            steps.append(("storage_journal.close", journal.close))

        errors: list[str] = []
        for name, fn in steps:
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                logger.warning("Memory agent shutdown step failed: %s", errors[-1])
        if errors:
            logger.warning("Memory agent shut down with %d error(s)", len(errors))

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict_conflicts(verdict_a: str, verdict_b: str) -> bool:
        positive = {"verified", "likely_verified"}
        negative = {"likely_hallucinated", "contradicted"}
        return (verdict_a in positive and verdict_b in negative) or (
            verdict_a in negative and verdict_b in positive
        )

    @staticmethod
    def _check_persistability(
        verdict: Optional[str],
        verification_status: Optional[str],
        confidence: Optional[float],
    ) -> tuple[bool, str]:
        """§13 gate shared by store_fact() and update_fact().

        A fact may only remain in factual memory when its verdict, verification
        status, and confidence are all supported. CONTRADICTED, UNCERTAIN,
        UNVERIFIED and negative verdicts must never be (re)persisted; updates
        that would push a stored fact into one of these states must remove it
        from factual memory instead.
        """
        v = str(verdict or "").strip().lower()
        if v in _BLOCKING_VERDICTS:
            return False, f"non_supported_verdict:{v}"

        verification_status = str(verification_status or "").strip().upper()
        if verification_status in _BLOCKING_STATUSES:
            return False, f"non_persistable_verification_status:{verification_status}"

        if confidence is None or confidence <= 0.0:
            return False, "zero_confidence_fact"

        return True, ""

    @staticmethod
    def _persistability_gate(request: StoreFactRequest) -> tuple[bool, str]:
        """Spec §13: only sufficiently-supported facts may become permanent memory.

        CONTRADICTED, NOT_ENOUGH_EVIDENCE, and UNCERTAIN pipeline states (as well
        as negative verdicts) must never be persisted as factual memory. Negative
        examples still feed the pattern learner separately.
        """
        return MemoryAgent._check_persistability(
            verdict=request.verdict,
            verification_status=request.verification_status,
            confidence=request.confidence,
        )

    async def store_fact(self, request: StoreFactRequest) -> StoreFactResponse:
        fact_id = str(uuid.uuid4())
        now = datetime.utcnow()
        contradictions: list[ContradictionAlert] = []

        # Duplicate + contradiction detection against existing memory. This runs
        # BEFORE the persistence gate so the system can still surface conflict
        # alerts about non-persistable input even though it is never stored.
        similar = self.vectors.search(
            query=request.claim_text,
            top_k=self._settings.contradiction_top_k,
            metadata_filter={"domain": request.domain} if request.domain else None,
        )
        duplicate_of: Optional[str] = None
        for res in similar:
            if (
                res.score >= self._settings.duplicate_similarity_threshold
                and res.metadata.get("verdict") == request.verdict
            ):
                duplicate_of = res.entry_id
            if (
                res.score >= self._settings.contradiction_similarity_threshold
                and self._verdict_conflicts(
                    request.verdict, str(res.metadata.get("verdict", ""))
                )
            ):
                # Stage 1 (vector similarity + verdict conflict) only found a
                # CANDIDATE. Stage 2 (NLI or structured comparison) confirms
                # whether the pair is an actual logical contradiction.
                confirmed, method, _detail = await self._confirm_contradiction(
                    existing_text=res.text,
                    new_text=request.claim_text,
                )
                if confirmed:
                    contradictions.append(
                        ContradictionAlert(
                            existing_fact_id=res.entry_id,
                            existing_claim_text=res.text,
                            similarity_score=res.score,
                            existing_verdict=str(res.metadata.get("verdict", "unknown")),
                            confirmation_method=method,
                            reason=(
                                f"New verdict '{request.verdict}' conflicts with stored "
                                f"verdict '{res.metadata.get('verdict')}' for similar "
                                f"claim (confirmed by {method})"
                            ),
                        )
                    )

        persistable, gate_reason = self._persistability_gate(request)
        if not persistable:
            pattern_updated = False
            if str(request.verdict or "").lower() in ("likely_hallucinated", "contradicted"):
                patterns = await self.patterns.observe_claim(
                    claim_text=request.claim_text,
                    domain=request.domain,
                    verdict=request.verdict,
                )
                pattern_updated = len(patterns) > 0
            logger.info(
                "Memory safety gate blocked persistence of %r: %s",
                request.claim_text[:60],
                gate_reason,
            )
            return StoreFactResponse(
                fact_id="",
                entities_created=0,
                edges_created=0,
                pattern_updated=pattern_updated,
                trust_updates=[],
                duplicate_of=duplicate_of,
                contradictions=contradictions,
                stored=False,
                reason=gate_reason,
            )

        if duplicate_of:
            logger.info("Duplicate claim detected, reusing fact %s", duplicate_of)
            return StoreFactResponse(
                fact_id=duplicate_of,
                entities_created=0,
                edges_created=0,
                pattern_updated=False,
                trust_updates=[],
                duplicate_of=duplicate_of,
                contradictions=contradictions,
                stored=False,
            )

        record_claim_text = request.claim_text
        kg_payload = {
            "fact_id": fact_id,
            "claim_text": request.claim_text,
            "domain": request.domain,
            "verdict": request.verdict,
            "confidence": request.confidence,
        }

        async with self._journal_context(
            "store", fact_id, "knowledge_graph", kg_payload,
            claim_text=record_claim_text, domain=request.domain,
        ):
            claim_node = self.kg.add_entity(
                name=request.claim_text[:120],
                entity_type=EntityType.CLAIM,
                properties={
                    "domain": request.domain,
                    "verdict": request.verdict,
                    "confidence": request.confidence,
                    "verification_status": request.verification_status,
                    "verified_at": request.verified_at.isoformat() if request.verified_at else now.isoformat(),
                    "provenance": request.provenance,
                    "origin": request.origin,
                    "fact_id": fact_id,
                },
                confidence=request.confidence,
            )

            fact_node = self.kg.add_entity(
                name=fact_id,
                entity_type=EntityType.FACT,
                properties={
                    "domain": request.domain,
                    "verdict": request.verdict,
                    "claim_text": request.claim_text,
                    "verification_status": request.verification_status,
                    "verified_at": request.verified_at.isoformat() if request.verified_at else now.isoformat(),
                    "provenance": request.provenance,
                    "origin": request.origin,
                },
                confidence=request.confidence,
            )
            self.kg.add_edge(
                source_id=claim_node.entity_id,
                target_id=fact_node.entity_id,
                relation=RelationType.DERIVED_FROM,
            )

            for evidence in request.evidence:
                source_id = evidence.get("source_id", "")
                if source_id:
                    source_node = self.kg.add_entity(
                        name=source_id,
                        entity_type=EntityType.SOURCE,
                        properties={"domain": request.domain},
                    )
                    self.kg.add_edge(
                        source_id=fact_node.entity_id,
                        target_id=source_node.entity_id,
                        relation=RelationType.MENTIONS,
                    )

            self.kg.save()

        entities_created = 2
        edges_created = 1 + sum(
            1 for e in request.evidence if e.get("source_id")
        )

        vector_payload = {
            "claim_text": request.claim_text,
            "metadata": {
                "domain": request.domain,
                "verdict": request.verdict,
                "confidence": request.confidence,
                "verification_status": request.verification_status,
                "verified_at": (
                    request.verified_at.isoformat()
                    if request.verified_at else now.isoformat()
                ),
                "provenance": request.provenance,
                "origin": request.origin,
                "fact_id": fact_id,
                "timestamp": now.isoformat(),
            },
        }
        async with self._journal_context(
            "store", fact_id, "vector_store", vector_payload,
            claim_text=record_claim_text, domain=request.domain,
        ):
            self.vectors.add(
                text=request.claim_text,
                metadata=vector_payload["metadata"],
                entry_id=fact_id,
            )

        pattern_updated = False
        if request.verdict in ("likely_hallucinated", "contradicted"):
            pattern_payload = {
                "claim_text": request.claim_text,
                "domain": request.domain,
                "verdict": request.verdict,
            }
            async with self._journal_context(
                "store", fact_id, "patterns", pattern_payload,
                claim_text=record_claim_text, domain=request.domain,
            ):
                patterns = await self.patterns.observe_claim(
                    claim_text=request.claim_text,
                    domain=request.domain,
                    verdict=request.verdict,
                )
                pattern_updated = len(patterns) > 0

        cache_payload = {
            "domain": request.domain,
            "claim_text": request.claim_text,
            "verdict": request.verdict,
            "confidence": request.confidence,
            "evidence_summary": f"Evidence from {len(request.evidence)} sources",
            "source_count": len(request.source_ids),
        }
        async with self._journal_context(
            "store", fact_id, "cache", cache_payload,
            claim_text=record_claim_text, domain=request.domain,
        ):
            await self.cache.set(
                domain=request.domain,
                claim_text=request.claim_text,
                verdict=request.verdict,
                evidence_summary=cache_payload["evidence_summary"],
                confidence=request.confidence,
                source_count=cache_payload["source_count"],
            )

        trust_updates: list[TrustUpdate] = []
        for sid in request.source_ids:
            reason = (
                TrustChangeReason.VERIFIED_CORRECT
                if request.verdict in ("verified", "likely_verified")
                else TrustChangeReason.VERIFIED_INCORRECT
            )
            trust_payload = {
                "source_id": sid,
                "domain": request.domain,
                "reason": reason.value,
                "evidence_count": len(request.evidence),
            }
            async with self._journal_context(
                "store", fact_id, "trust", trust_payload,
                claim_text=record_claim_text, domain=request.domain,
            ):
                update = await self.trust.update_trust(
                    source_id=sid,
                    source_name=sid,
                    domain=request.domain,
                    reason=reason,
                    evidence_count=len(request.evidence),
                )
                trust_updates.append(update)

        return StoreFactResponse(
            fact_id=fact_id,
            entities_created=entities_created,
            edges_created=edges_created,
            pattern_updated=pattern_updated,
            trust_updates=trust_updates,
            duplicate_of=None,
            contradictions=contradictions,
            stored=True,
        )

    async def store_facts_batch(
        self, requests: list[StoreFactRequest]
    ) -> BatchStoreResponse:
        results: list[StoreFactResponse] = []
        errors: list[dict[str, str]] = []
        stored = duplicates = failed = skipped = 0

        for request in requests:
            try:
                response = await self.store_fact(request)
                results.append(response)
                if response.duplicate_of:
                    duplicates += 1
                elif response.stored:
                    stored += 1
                elif response.reason:
                    skipped += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                errors.append(
                    {"claim_text": request.claim_text[:120], "error": str(e)}
                )

        return BatchStoreResponse(
            total=len(requests),
            stored=stored,
            duplicates=duplicates,
            failed=failed,
            skipped=skipped,
            results=results,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Stage-2 contradiction confirmation + storage journal
    # ------------------------------------------------------------------

    async def _confirm_contradiction(
        self, existing_text: str, new_text: str
    ) -> tuple[bool, Optional[str], str]:
        """Confirm a FAISS candidate pair as a real contradiction.

        Vector similarity only SURFACES candidates; this second stage decides.
        Returns (confirmed, method, detail) where method is 'nli' or
        'structured'. Candidates that neither method confirms are dropped.
        """
        loop = asyncio.get_running_loop()
        try:
            confirmed, method = await loop.run_in_executor(
                None, self.contradictions.confirm, existing_text, new_text
            )
        except Exception as e:
            logger.warning("Stage-2 contradiction check failed: %s", e)
            return False, None, "stage2_error"
        detail = f"confirmed by {method}" if confirmed else "unconfirmed"
        return confirmed, method, detail

    async def _journal(
        self,
        op_type: str,
        fact_id: str,
        subsystem: str,
        status: str = "pending",
        payload: Optional[dict[str, Any]] = None,
        claim_text: Optional[str] = None,
        domain: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        try:
            if self.journal is not None:
                return await self.journal.record(
                    op_type=op_type,
                    fact_id=fact_id,
                    subsystem=subsystem,
                    status=status,
                    payload=payload,
                    claim_text=claim_text,
                    domain=domain,
                    error=error,
                )
        except Exception as e:
            logger.warning("Storage journal write failed: %s", e)
        return ""

    @asynccontextmanager
    async def _journal_context(
        self,
        op_type: str,
        fact_id: str,
        subsystem: str,
        payload: dict[str, Any],
        claim_text: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Record a per-subsystem outcome and isolate subsystem failures.

        A single fact write spans Knowledge Graph, FAISS, Pattern DB, Cache DB,
        and Source Trust DB — there is no distributed transaction across them.
        Each write is journaled 'pending' before, then marked 'done' or
        'failed' after. A failure is isolated (not re-raised) so the other
        subsystems still complete, and the failed operation can be reconciled
        later from its journal payload.
        """
        op_id = await self._journal(
            op_type, fact_id, subsystem, "pending", payload,
            claim_text=claim_text, domain=domain,
        )
        try:
            yield
        except Exception as e:
            logger.warning(
                "Subsystem write failed (%s, fact %s): %s",
                subsystem, fact_id, e,
            )
            if op_id:
                try:
                    await self.journal.mark_failed(op_id, str(e))
                except Exception as je:
                    logger.warning("Failed to journal failure: %s", je)
        else:
            if op_id:
                try:
                    await self.journal.mark_done(op_id)
                except Exception as je:
                    logger.warning("Failed to journal success: %s", je)

    async def reconcile(self, limit: int = 50) -> dict[str, Any]:
        """Retry failed subsystem writes for derived indexes.

        knowledge_graph is the canonical record and is not auto-rebuilt;
        vector_store and cache are derived indexes reconstructed from the
        journal payload. Patterns/trust failures from the retry are surfaced as
        unresolved and require manual review.
        """
        failed_ops = await self.journal.get_failed(limit)
        reconciled = unresolved = 0
        errors: list[dict[str, str]] = []
        for op in failed_ops:
            try:
                if await self._retry_operation(op):
                    await self.journal.mark_reconciled(op["op_id"])
                    reconciled += 1
                else:
                    unresolved += 1
            except Exception as e:
                unresolved += 1
                errors.append({"op_id": op["op_id"], "error": str(e)})

        return {
            "attempted": len(failed_ops),
            "reconciled": reconciled,
            "unresolved": unresolved,
            "errors": errors,
        }

    async def _retry_operation(self, op: dict[str, Any]) -> bool:
        subsystem = op["subsystem"]
        payload = op.get("payload") or {}
        fact_id = op["fact_id"]

        if subsystem == "vector_store":
            claim_text = payload.get("claim_text") or op.get("claim_text")
            metadata = payload.get("metadata") or {}
            if not claim_text:
                return False
            if self.vectors.get(fact_id) is not None:
                return True  # already present
            self.vectors.add(
                text=claim_text,
                metadata=metadata,
                entry_id=fact_id,
            )
            return True

        if subsystem == "cache":
            claim_text = payload.get("claim_text") or op.get("claim_text")
            domain = payload.get("domain") or op.get("domain") or "general"
            if not claim_text:
                return False
            await self.cache.set(
                domain=domain,
                claim_text=claim_text,
                verdict=payload.get("verdict", "verified"),
                evidence_summary=payload.get(
                    "evidence_summary", "Recovered by storage reconciliation"
                ),
                confidence=float(payload.get("confidence", 0.5)),
                source_count=int(payload.get("source_count", 0)),
            )
            return True

        if subsystem == "patterns":
            claim_text = payload.get("claim_text") or op.get("claim_text")
            domain = payload.get("domain") or op.get("domain")
            verdict = payload.get("verdict")
            if not claim_text or not domain or not verdict:
                return False
            await self.patterns.observe_claim(
                claim_text=claim_text,
                domain=domain,
                verdict=verdict,
            )
            return True

        if subsystem == "trust":
            source_id = payload.get("source_id")
            domain = payload.get("domain")
            reason_value = payload.get("reason")
            if not source_id or not domain or not reason_value:
                return False
            try:
                reason = TrustChangeReason(reason_value)
            except ValueError:
                return False
            await self.trust.update_trust(
                source_id=source_id,
                source_name=source_id,
                domain=domain,
                reason=reason,
                evidence_count=int(payload.get("evidence_count", 1)),
            )
            return True

        # knowledge_graph is the canonical source of truth; it is NOT rebuilt
        # automatically from a partial payload.
        logger.warning(
            "Reconciliation for subsystem %r requires manual review", subsystem,
        )
        return False

    async def get_storage_journal_stats(self) -> dict[str, Any]:
        return await self.journal.get_stats()

    # ------------------------------------------------------------------
    # Fact lifecycle
    # ------------------------------------------------------------------

    async def delete_fact(self, fact_id: str) -> DeleteFactResponse:
        """Remove a fact from all subsystems (KG, vectors, cache)."""
        deleted_from: list[str] = []

        # Knowledge Graph — fact_id is stored as entity NAME, not ID
        entities = self.kg.find_entity_by_name(fact_id, EntityType.FACT)
        entity = entities[0] if entities else None
        if entity:
            async with self._journal_context(
                "delete", fact_id, "knowledge_graph", {"fact_id": fact_id},
            ):
                self.kg.remove_entity(entity.entity_id)
            deleted_from.append("knowledge_graph")

        # Vector Store
        async with self._journal_context(
            "delete", fact_id, "vector_store", {"fact_id": fact_id},
        ):
            deleted_vector = self.vectors.delete(fact_id)
            if not deleted_vector:
                raise RuntimeError("vector entry missing for delete")
        if deleted_vector:
            deleted_from.append("vector_store")

        # Cache — invalidate by claim text if found in KG properties
        if entity and entity.properties.get("claim_text"):
            domain = entity.properties.get("domain", "general")
            claim = entity.properties["claim_text"]
            async with self._journal_context(
                "delete", fact_id, "cache",
                {"domain": domain, "claim_text": claim},
                claim_text=claim, domain=domain,
            ):
                removed = await self.cache.invalidate(domain, claim)
                if not removed:
                    raise RuntimeError("cache entry missing for delete")
            if removed:
                deleted_from.append("cache")

        self.kg.save()

        return DeleteFactResponse(
            fact_id=fact_id,
            deleted=len(deleted_from) > 0,
            deleted_from=deleted_from,
        )

    async def update_fact(self, request: UpdateFactRequest) -> UpdateFactResponse:
        """Update verdict/confidence of an existing fact across subsystems.

        An update must pass the same §13 persistability gate as a fresh store.
        If the new state is non-persistable (e.g. a verified fact becoming
        contradicted), the fact is REMOVED from factual memory rather than
        remaining behind as a contradicting record. The retraction is still
        reported so the caller can decide whether to alert on it.
        """
        # fact_id is stored as entity NAME, not ID
        entities = self.kg.find_entity_by_name(request.fact_id, EntityType.FACT)
        entity = entities[0] if entities else None
        if not entity:
            raise ValueError(f"Fact {request.fact_id} not found")

        old_verdict = str(entity.properties.get("verdict", "unknown"))
        old_confidence = float(entity.properties.get("confidence", 0.0))
        new_verdict = request.new_verdict or old_verdict
        new_confidence = (
            request.new_confidence
            if request.new_confidence is not None
            else old_confidence
        )
        verification_status = str(
            entity.properties.get("verification_status", "VERIFIED")
        ).strip().upper()

        persistable, gate_reason = MemoryAgent._check_persistability(
            verdict=new_verdict,
            verification_status=verification_status,
            confidence=new_confidence,
        )
        if not persistable:
            logger.info(
                "Update would invalidate factual memory, removing fact %s: %s",
                request.fact_id,
                gate_reason,
            )
            deleted = await self.delete_fact(request.fact_id)
            removed_from = list(deleted.deleted_from or [])

            # Record the retraction as pattern history so it is not lost.
            if str(new_verdict).lower() in ("likely_hallucinated", "contradicted"):
                claim_text = entity.properties.get("claim_text", "")
                if claim_text:
                    try:
                        await self.patterns.observe_claim(
                            claim_text=claim_text,
                            domain=str(entity.properties.get("domain", "general")),
                            verdict=new_verdict,
                        )
                    except Exception as e:
                        logger.warning("Failed to record retraction pattern: %s", e)

            return UpdateFactResponse(
                fact_id=request.fact_id,
                old_verdict=old_verdict,
                new_verdict=new_verdict,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                updated_in=[],
                persisted=False,
                removed_from=removed_from,
                reason=gate_reason,
            )

        updated_in: list[str] = []

        # Update KG entity properties
        async with self._journal_context(
            "update", request.fact_id, "knowledge_graph",
            {"fact_id": request.fact_id, "new_verdict": new_verdict,
             "new_confidence": new_confidence},
        ):
            entity.properties["verdict"] = new_verdict
            entity.properties["confidence"] = new_confidence
            entity.updated_at = datetime.utcnow()
            updated_in.append("knowledge_graph")
            self.kg.save()

        # Update vector store metadata
        async with self._journal_context(
            "update", request.fact_id, "vector_store",
            {"fact_id": request.fact_id, "new_verdict": new_verdict,
             "new_confidence": new_confidence},
        ):
            vec_entry = self.vectors.get(request.fact_id)
            if vec_entry and vec_entry.metadata:
                vec_entry.metadata["verdict"] = new_verdict
                vec_entry.metadata["confidence"] = new_confidence
                updated_in.append("vector_store")
            else:
                raise RuntimeError("vector entry missing for fact")

        # Update cache if claim text exists
        claim_text = entity.properties.get("claim_text")
        if claim_text:
            domain = entity.properties.get("domain", "general")
            async with self._journal_context(
                "update", request.fact_id, "cache",
                {"domain": domain, "claim_text": claim_text,
                 "verdict": new_verdict, "confidence": new_confidence},
                claim_text=claim_text, domain=domain,
            ):
                # Invalidate old cache, re-set with new verdict
                await self.cache.invalidate(domain, claim_text)
                await self.cache.set(
                    domain=domain,
                    claim_text=claim_text,
                    verdict=new_verdict,
                    evidence_summary=f"Updated: {new_verdict}",
                    confidence=new_confidence,
                    source_count=0,
                )
                updated_in.append("cache")

        return UpdateFactResponse(
            fact_id=request.fact_id,
            old_verdict=old_verdict,
            new_verdict=new_verdict,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            updated_in=updated_in,
            persisted=True,
        )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(self, request: RecallRequest) -> RecallResponse:
        cached_verification: Optional[CachedVerification] = None
        fuzzy_cache_hits: list[CachedVerification] = []
        similar_facts: list[VectorSearchResult] = []
        related_entities = []
        relevant_patterns: list[HallucinationPattern] = []
        source_trust: list[SourceTrustRecord] = []

        # Graceful degradation: each subsystem failure is isolated
        if request.include_cache:
            try:
                cached_verification = await self.cache.get(
                    domain=request.domain or "general",
                    claim_text=request.query,
                )
            except Exception as e:
                logger.warning("Cache lookup failed, continuing: %s", e)

        try:
            similar_facts = self.vectors.search(
                query=request.query,
                top_k=request.top_k,
                metadata_filter={"domain": request.domain} if request.domain else None,
            )
        except Exception as e:
            logger.warning("Vector search failed, continuing: %s", e)

        # Fuzzy cache: near-duplicate claims that were verified before
        if request.include_cache and request.fuzzy_cache:
            for fact in similar_facts:
                if fact.score >= self._settings.fuzzy_cache_threshold:
                    try:
                        hit = await self.cache.get(
                            domain=request.domain or "general",
                            claim_text=fact.text,
                        )
                    except Exception:
                        hit = None
                    if hit and hit.cache_key not in {
                        h.cache_key for h in fuzzy_cache_hits
                    }:
                        fuzzy_cache_hits.append(hit)

        if request.include_graph_context:
            try:
                claim_nodes = self.kg.find_entity_by_name(request.query[:120])
                for node in claim_nodes[:5]:
                    neighbors = self.kg.get_neighbors(node.entity_id)
                    for neighbor, edge in neighbors:
                        neighbor.access_count += 1
                        related_entities.append(neighbor)
            except Exception as e:
                logger.warning("Graph lookup failed, continuing: %s", e)

        if request.include_patterns:
            try:
                relevant_patterns = await self.patterns.query_patterns(
                    domain=request.domain,
                    top_k=10,
                )
            except Exception as e:
                logger.warning("Pattern query failed, continuing: %s", e)

        if request.domain:
            try:
                source_trust = await self.trust.get_domain_sources(
                    domain=request.domain,
                    min_trust=0.3,
                )
            except Exception as e:
                logger.warning("Trust lookup failed, continuing: %s", e)

        reranked = False
        if request.rerank and similar_facts:
            try:
                similar_facts = await self._rerank_results(similar_facts, request.domain)
                reranked = True
            except Exception as e:
                logger.warning("Reranking failed, using raw scores: %s", e)

        if request.min_similarity > 0:
            similar_facts = [
                f for f in similar_facts
                if f.score >= request.min_similarity
            ]

        return RecallResponse(
            cached_verification=cached_verification,
            fuzzy_cache_hits=fuzzy_cache_hits,
            similar_facts=similar_facts,
            related_entities=related_entities,
            relevant_patterns=relevant_patterns,
            source_trust=source_trust,
            reranked=reranked,
        )

    async def _rerank_results(
        self,
        results: list[VectorSearchResult],
        domain: Optional[str],
    ) -> list[VectorSearchResult]:
        """Blend vector similarity with source trust and recency."""
        # Fetch domain trust once, not per result
        domain_trust_avg = 0.5
        if domain:
            domain_sources = await self.trust.get_domain_sources(
                domain=domain, min_trust=0.0
            )
            if domain_sources:
                domain_trust_avg = sum(
                    s.trust_score for s in domain_sources
                ) / len(domain_sources)

        now = datetime.utcnow()
        reranked: list[VectorSearchResult] = []
        for result in results:
            # Recency score
            try:
                ts = result.metadata.get("timestamp")
                if ts:
                    age_days = (now - datetime.fromisoformat(ts)).total_seconds() / 86400
                    recency = max(0.0, 1.0 - age_days / 365.0)
                else:
                    recency = 0.5
            except Exception:
                recency = 0.5

            blended = (
                self._settings.rerank_alpha * result.score
                + self._settings.rerank_beta * domain_trust_avg
                + self._settings.rerank_gamma * recency
            )
            reranked.append(
                VectorSearchResult(
                    entry_id=result.entry_id,
                    text=result.text,
                    score=result.score,
                    metadata=result.metadata,
                    reranked_score=round(min(1.0, blended), 4),
                )
            )

        reranked.sort(key=lambda r: r.reranked_score or 0.0, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    async def check_cache(
        self, domain: str, claim_text: str
    ) -> Optional[dict]:
        result = await self.cache.get(domain, claim_text)
        return result.model_dump() if result else None

    async def invalidate_cache(self, domain: str, claim_text: str) -> bool:
        return await self.cache.invalidate(domain, claim_text)

    # ------------------------------------------------------------------
    # Trust operations
    # ------------------------------------------------------------------

    async def update_source_trust(
        self,
        source_id: str,
        source_name: str,
        domain: str,
        reason: TrustChangeReason,
        evidence_count: int = 1,
    ) -> TrustUpdate:
        return await self.trust.update_trust(
            source_id=source_id,
            source_name=source_name,
            domain=domain,
            reason=reason,
            evidence_count=evidence_count,
        )

    async def get_source_trust(self, source_id: str) -> Optional[SourceTrustRecord]:
        return await self.trust.get_trust(source_id)

    # ------------------------------------------------------------------
    # Pattern operations
    # ------------------------------------------------------------------

    async def query_patterns(
        self,
        domain: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> list[HallucinationPattern]:
        return await self.patterns.query_patterns(
            domain=domain,
            pattern_type=pattern_type,
            min_confidence=min_confidence,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> MemoryStatsResponse:
        kg_stats = self.kg.get_stats()
        cache_stats = await self.cache.get_stats()
        patterns_total = await self.patterns.get_total_count()
        sources_total = await self.trust.get_total_sources()
        vector_size = self.vectors.size()

        return MemoryStatsResponse(
            knowledge_graph=kg_stats,
            cache=cache_stats,
            patterns_total=patterns_total,
            sources_tracked=sources_total,
            vector_store_size=vector_size,
        )

    async def save_all(self) -> None:
        self.kg.save()
        self.vectors.save()


_memory_agent_instance: Optional[MemoryAgent] = None


def get_memory_agent(settings: Optional[Settings] = None) -> MemoryAgent:
    global _memory_agent_instance
    if _memory_agent_instance is None:
        _memory_agent_instance = MemoryAgent(settings=settings)
    return _memory_agent_instance
