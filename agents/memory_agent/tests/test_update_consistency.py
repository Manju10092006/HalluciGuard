"""Real isolated JSON/SQLite persistence; mocked failures, no embedding model."""
import json
import os
from types import SimpleNamespace

import pytest

from agents.memory_agent.cache.verification_cache import VerificationCache
from agents.memory_agent.config.settings import Settings
from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
from agents.memory_agent.memory.memory_agent import MemoryAgent, MemoryConsistencyError
from agents.memory_agent.schemas.models import EntityType, RelationType, UpdateFactRequest, VectorEntry
from agents.memory_agent.vector_store.faiss_store import VectorStore


@pytest.fixture
async def memory(tmp_path):
    settings = Settings(kg_persistence_path=str(tmp_path / "kg.json"),
                        cache_db_path=str(tmp_path / "cache.db"),
                        vector_store_path=str(tmp_path / "vectors"))
    kg = KnowledgeGraph(settings.kg_persistence_path)
    claim = kg.add_entity("Fixture claim.", EntityType.CLAIM,
        {"fact_id": "fixture-id", "verdict": "verified", "confidence": .8}, confidence=.8)
    fact = kg.add_entity("fixture-id", EntityType.FACT,
        {"claim_text": "Fixture claim.", "domain": "test", "verdict": "verified"}, confidence=.8)
    kg.add_edge(claim.entity_id, fact.entity_id, RelationType.DERIVED_FROM)
    kg.save()
    cache = VerificationCache(settings.cache_db_path)
    await cache.initialize()
    await cache.set("test", "Fixture claim.", "verified", "fixture", .8, 1)
    vectors = VectorStore(store_path=settings.vector_store_path)
    # Lifecycle updates require metadata, not embeddings or model inference.
    vectors._entries = [VectorEntry(entry_id="fixture-id", text="Fixture claim.",
        metadata={"fact_id": "fixture-id", "verdict": "verified", "confidence": .8})]
    vectors._id_map = {"fixture-id": 0}
    vectors.save()
    agent = MemoryAgent(settings=settings, knowledge_graph=kg, cache=cache, vector_store=vectors)
    yield SimpleNamespace(agent=agent, claim=claim, fact=fact, path=tmp_path)
    await cache.close()


def request(verdict="unverified", confidence=0):
    return UpdateFactRequest(fact_id="fixture-id", new_verdict=verdict, new_confidence=confidence)


async def assert_durable(memory, verdict="unverified", confidence=0):
    rows = json.loads((memory.path / "kg.json").read_text())["nodes"]
    assert len(rows) == 2
    for row in rows:
        assert row["properties"]["verdict"] == verdict
        assert row["properties"]["confidence"] == confidence
        assert row["confidence"] == confidence
        assert memory.agent.kg._graph.nodes[row["entity_id"]]["confidence"] == confidence
    row = json.loads((memory.path / "vectors" / "entries.json").read_text())[0]
    assert row["metadata"]["verdict"] == verdict
    assert row["metadata"]["confidence"] == confidence
    async with memory.agent.cache._db.execute("SELECT verdict, confidence FROM verification_cache") as cursor:
        assert await cursor.fetchall() == [(verdict, confidence)]


@pytest.mark.asyncio
async def test_quarantine_is_durable_idempotent_and_does_not_touch_index(memory):
    index = memory.path / "vectors" / "index.faiss"
    index.write_bytes(b"index-is-not-rebuilt")
    for _ in range(2):
        response = await memory.agent.update_fact(request())
        assert set(response.updated_in) == {"knowledge_graph", "vector_store", "cache"}
        await assert_durable(memory)
        assert index.read_bytes() == b"index-is-not-rebuilt"


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["knowledge_graph", "vector_store", "cache"])
async def test_partial_write_attempts_other_stores_and_retry_repairs(memory, monkeypatch, store):
    def fail(*args, **kwargs):
        raise OSError("dummy-private-secret")
    async def fail_async(*args, **kwargs):
        fail()
    with monkeypatch.context() as patch:
        if store == "knowledge_graph":
            patch.setattr(memory.agent.kg, "save", fail)
        elif store == "vector_store":
            # Failure after in-memory mutation but before durable replacement.
            patch.setattr("agents.memory_agent.vector_store.faiss_store.os",
                          SimpleNamespace(replace=fail, fsync=os.fsync))
        else:
            patch.setattr(memory.agent.cache, "set", fail_async)
        with pytest.raises(MemoryConsistencyError) as caught:
            await memory.agent.update_fact(request())
        error = caught.value
        assert store + ".write" in error.failures
        assert "dummy-private-secret" not in str(error) + str(error.failures)
        assert set(error.updated_in) == {"knowledge_graph", "vector_store", "cache"} - {store}
        assert memory.fact.confidence == memory.claim.confidence == 0
    await memory.agent.update_fact(request())
    await assert_durable(memory)


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["kg", "vectors", "cache"])
async def test_readback_failure_is_never_success(memory, monkeypatch, store):
    def fail(*args):
        raise OSError("dummy-private-secret")
    async def fail_async(*args):
        fail()
    with monkeypatch.context() as patch:
        patch.setattr(getattr(memory.agent, store), "verify_persisted_verification",
                      fail_async if store == "cache" else fail)
        with pytest.raises(MemoryConsistencyError) as caught:
            await memory.agent.update_fact(request())
        assert any(k.endswith(".readback") for k in caught.value.failures)
    await assert_durable(memory)
    await memory.agent.update_fact(request())
    await assert_durable(memory)


@pytest.mark.asyncio
async def test_silent_graph_save_failure_detected_by_independent_readback(memory, monkeypatch):
    monkeypatch.setattr(memory.agent.kg, "save", lambda: None)
    with pytest.raises(MemoryConsistencyError) as caught:
        await memory.agent.update_fact(request())
    assert "knowledge_graph.readback" in caught.value.failures


@pytest.mark.asyncio
async def test_missing_vector_is_explicit_failure_but_other_stores_quarantine(memory):
    memory.agent.vectors._id_map.clear()
    with pytest.raises(MemoryConsistencyError) as caught:
        await memory.agent.update_fact(request())
    assert "vector_store.write" in caught.value.failures
    assert set(caught.value.updated_in) == {"knowledge_graph", "cache"}
    memory.agent.vectors._id_map["fixture-id"] = 0
    await memory.agent.update_fact(request())
    await assert_durable(memory)


@pytest.mark.asyncio
async def test_normal_update_and_missing_confidence_preserve_contract(memory):
    response = await memory.agent.update_fact(request("likely_hallucinated", .2))
    assert response.old_confidence == .8  # legacy fact properties lacked confidence
    await assert_durable(memory, "likely_hallucinated", .2)
    response = await memory.agent.update_fact(request("verified", None))
    assert response.new_confidence == .2
    await assert_durable(memory, "verified", .2)


@pytest.mark.asyncio
async def test_ambiguous_claim_ownership_stops_before_any_write(memory):
    memory.claim.properties["fact_id"] = "other-id"
    with pytest.raises(ValueError, match="ambiguous"):
        await memory.agent.update_fact(request())
    assert memory.fact.properties["verdict"] == "verified"
    assert memory.agent.vectors.get("fixture-id").metadata["verdict"] == "verified"


@pytest.mark.asyncio
async def test_networkx_and_disk_confidence_consistent_after_reload(memory):
    await memory.agent.update_fact(request())
    loaded = KnowledgeGraph(str(memory.path / "kg.json"))
    for node in loaded.verification_nodes("fixture-id"):
        assert loaded._graph.nodes[node.entity_id]["confidence"] == node.confidence == 0


@pytest.mark.asyncio
async def test_retry_repairs_legacy_partial_quarantine(memory):
    # Exact defect: fact properties, vector and cache safe, claim/top-level stale.
    memory.fact.properties.update(verdict="unverified", confidence=0)
    memory.agent.kg.save()
    memory.agent.vectors.update_verification("fixture-id", "unverified", 0)
    await memory.agent.cache.set("test", "Fixture claim.", "unverified", "fixture", 0, 0)
    response = await memory.agent.update_fact(request())
    assert response.old_confidence == response.new_confidence == 0
    await assert_durable(memory)


@pytest.mark.asyncio
async def test_silent_vector_write_is_detected(memory, monkeypatch):
    monkeypatch.setattr(memory.agent.vectors, "update_verification", lambda *args: None)
    with pytest.raises(MemoryConsistencyError) as caught:
        await memory.agent.update_fact(request())
    assert "vector_store.readback" in caught.value.failures
