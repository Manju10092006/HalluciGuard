"""Memory contract regressions using injected storage, without vector models."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.memory_agent.memory.memory_agent import MemoryAgent
from agents.memory_agent.schemas.models import StoreFactRequest, UpdateFactRequest


def evidence(label="entailment"):
    item = {"source_id": "fixture", "snippet": "The launch occurred in 2020."}
    if label is not None:
        item["entailment_label"] = label
    return item


def storage_agent(similar=()):
    agent = object.__new__(MemoryAgent)
    agent._settings = SimpleNamespace(
        contradiction_top_k=5,
        duplicate_similarity_threshold=.95,
        contradiction_similarity_threshold=.75,
    )
    agent.kg = MagicMock()
    agent.kg.add_entity.side_effect = [
        SimpleNamespace(entity_id="claim"), SimpleNamespace(entity_id="fact"),
        SimpleNamespace(entity_id="source"),
    ]
    agent.vectors = MagicMock()
    agent.vectors.search.return_value = list(similar)
    agent.cache = SimpleNamespace(set=AsyncMock(), invalidate=AsyncMock())
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("label", [None, "neutral", "contradiction", "supports", "unknown"])
async def test_verified_store_requires_explicit_entailment_before_storage(label):
    agent = object.__new__(MemoryAgent)
    with pytest.raises(ValueError, match="requires_evidence_and_provenance_with_entailment"):
        await agent.store_fact(StoreFactRequest(
            claim_text="The launch occurred in 2020.", domain="general",
            verdict="verified", confidence=.9, evidence=[evidence(label)],
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize("label", [None, "neutral", "contradiction"])
async def test_promotion_requires_directional_support_before_any_mutation(label):
    agent = object.__new__(MemoryAgent)
    properties = {"verdict": "contradicted", "confidence": .8, "evidence": []}
    fact = SimpleNamespace(properties=properties)
    agent.kg = MagicMock()
    agent.kg.find_entity_by_name.return_value = [fact]
    with pytest.raises(ValueError, match="requires_evidence_and_provenance_with_entailment"):
        await agent.update_fact(UpdateFactRequest(
            fact_id="existing", new_verdict="verified", new_evidence=[evidence(label)],
        ))
    assert properties == {"verdict": "contradicted", "confidence": .8, "evidence": []}
    agent.kg.save.assert_not_called()


@pytest.mark.asyncio
async def test_verified_store_preserves_support_and_normalizes_verdict():
    agent = storage_agent()
    result = await agent.store_fact(StoreFactRequest(
        claim_text="The launch occurred in 2020.", domain="general",
        verdict=" VERIFIED ", confidence=.9, evidence=[evidence()],
    ))
    assert result.stored
    metadata = agent.vectors.add.call_args.kwargs["metadata"]
    assert metadata["verdict"] == "verified"
    assert metadata["evidence"] == [evidence()]
    assert agent.cache.set.await_args.kwargs["verdict"] == "verified"
    agent.vectors.save.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_text", [
    "The launch occurred in 2021.", "The launch did not occur in 2020.",
])
async def test_semantic_neighbor_cannot_replace_a_distinct_fact(new_text):
    neighbor = SimpleNamespace(
        text="The launch occurred in 2020.", score=.999, entry_id="existing",
        metadata={"verdict": "verified"},
    )
    agent = storage_agent([neighbor])
    support = {**evidence(), "snippet": new_text}
    result = await agent.store_fact(StoreFactRequest(
        claim_text=new_text, domain="general", verdict="verified",
        confidence=.9, evidence=[support],
    ))
    assert result.stored
    assert result.duplicate_of is None
    assert result.fact_id != neighbor.entry_id
    agent.vectors.add.assert_called_once()


@pytest.mark.asyncio
async def test_exact_claim_still_reuses_existing_fact():
    neighbor = SimpleNamespace(
        text="The launch occurred in 2020.", score=.999, entry_id="existing",
        metadata={"verdict": "verified"},
    )
    agent = storage_agent([neighbor])
    result = await agent.store_fact(StoreFactRequest(
        claim_text="The launch occurred in 2020.", domain="general",
        verdict="verified", confidence=.9, evidence=[evidence()],
    ))
    assert result.duplicate_of == "existing"
    assert not result.stored
    agent.vectors.add.assert_not_called()
    agent.kg.add_entity.assert_not_called()
