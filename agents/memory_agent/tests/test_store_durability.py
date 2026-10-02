"""Real isolated JSON/SQLite stores, existing mock embedding fixture."""
import json
import asyncio

import pytest

from agents.memory_agent.tests.test_memory_agent import agent
from agents.memory_agent.tests.test_update_consistency import memory, request, assert_durable
from agents.memory_agent.schemas.models import StoreFactRequest
from agents.memory_agent.memory.memory_agent import MemoryConsistencyError


@pytest.mark.asyncio
async def test_first_fact_is_durable_before_close(agent):
    result = await agent.store_fact(StoreFactRequest(claim_text="Durability contract fixture.",
                                                   domain="test", verdict="verified", confidence=.8))
    assert result.stored
    from pathlib import Path
    entries = json.loads((Path(agent._settings.vector_store_path) / "entries.json").read_text())
    assert any(r["entry_id"] == result.fact_id for r in entries)
    nodes = agent.kg.verification_nodes(result.fact_id)
    agent.kg.verify_persisted_verification(nodes, "verified", .8)
    agent.vectors.verify_persisted_verification(result.fact_id, "verified", .8)
    await agent.cache.verify_persisted_verification("test", "Durability contract fixture.", "verified", .8)


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["kg", "vectors"])
async def test_failed_or_silent_save_never_reports_stored(agent, monkeypatch, store):
    monkeypatch.setattr(getattr(agent, store), "save", lambda: None)
    with pytest.raises(MemoryConsistencyError) as caught:
        await agent.store_fact(StoreFactRequest(claim_text="Unsaved fixture.", domain="test",
                                               verdict="verified", confidence=.8))
    assert any(k.endswith(".readback") for k in caught.value.failures)
    assert caught.value.fact_id


@pytest.mark.asyncio
async def test_write_failure_is_classified_without_disclosing_provider_text(agent, monkeypatch):
    def fail(): raise OSError("dummy-secret-storage-path")
    monkeypatch.setattr(agent.vectors, "save", fail)
    with pytest.raises(MemoryConsistencyError) as caught:
        await agent.store_fact(StoreFactRequest(claim_text="Failed-save fixture.", domain="test",
                                               verdict="verified", confidence=.8))
    assert caught.value.failures["vector_store.write"] == "OSError"
    assert "dummy-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_concurrent_quarantine_requests_on_one_agent_remain_durable(memory):
    outcomes = await asyncio.gather(*(memory.agent.update_fact(request()) for _ in range(20)))
    assert all(r.new_confidence == 0 and r.new_verdict == "unverified" for r in outcomes)
    await assert_durable(memory)


@pytest.mark.asyncio
async def test_partial_store_cannot_be_reused_as_a_successful_duplicate(agent, monkeypatch):
    request = StoreFactRequest(claim_text="Partial store fixture.", domain="test",
                               verdict="verified", confidence=.8)
    with monkeypatch.context() as patch:
        patch.setattr(agent.vectors, "save", lambda: None)
        with pytest.raises(MemoryConsistencyError):
            await agent.store_fact(request)
    with pytest.raises(MemoryConsistencyError) as caught:
        await agent.store_fact(request)
    assert "vector_store.readback" in caught.value.failures
    # Caller can repair the explicitly identified partial fact, never silently
    # promoting an inaccessible representation through the duplicate shortcut.
    from agents.memory_agent.schemas.models import UpdateFactRequest
    await agent.update_fact(UpdateFactRequest(fact_id=caught.value.fact_id,
                                             new_verdict="verified", new_confidence=.8))
    duplicate = await agent.store_fact(request)
    assert not duplicate.stored and duplicate.duplicate_of == caught.value.fact_id


@pytest.mark.asyncio
async def test_duplicate_reuses_existing_confidence_without_promoting_it(agent):
    original = await agent.store_fact(StoreFactRequest(claim_text="Stable duplicate fixture.",
        domain="test", verdict="verified", confidence=.6))
    duplicate = await agent.store_fact(StoreFactRequest(claim_text="Stable duplicate fixture.",
        domain="test", verdict="verified", confidence=.9))
    assert duplicate.duplicate_of == original.fact_id and not duplicate.stored
    agent.vectors.verify_persisted_verification(original.fact_id, "verified", .6)


@pytest.mark.asyncio
async def test_verified_legacy_duplicate_can_reuse_real_top_level_confidence(agent):
    original = await agent.store_fact(StoreFactRequest(claim_text="Legacy fixture.",
        domain="test", verdict="verified", confidence=.6))
    nodes = agent.kg.verification_nodes(original.fact_id)
    nodes[0].properties.pop("confidence")
    agent.kg.save()
    duplicate = await agent.store_fact(StoreFactRequest(claim_text="Legacy fixture.",
        domain="test", verdict="verified", confidence=.9))
    assert duplicate.duplicate_of == original.fact_id
    assert nodes[0].confidence == .6
    # Ordinary reuse compatibility must not weaken quarantine's strict check.
    agent.kg.set_verification(nodes, "unverified", 0)
    nodes[0].properties.pop("confidence")
    agent.kg.save()
    with pytest.raises(RuntimeError, match="read-back"):
        agent.kg.verify_persisted_verification(nodes, "unverified", 0,
                                             allow_legacy_verified_confidence=True)
