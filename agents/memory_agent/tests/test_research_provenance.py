import json
import pytest
from .test_memory_agent import agent
from agents.memory_agent.schemas.models import StoreFactRequest, EntityType

@pytest.mark.asyncio
async def test_provenance_survives_all_storage_records(agent):
    evidence=[{'source_id':'reference','title':'Reference card','url':'https://reference.example/a','snippet':'A is located in B.'}]
    request=StoreFactRequest(claim_text='A is located in B.',domain='general',verdict='verified',
                            evidence=evidence,source_ids=['reference'],confidence=.9,metadata={'execution_id':'run-1'})
    result=await agent.store_fact(request)
    graph=agent.kg.find_entity_by_name(result.fact_id,EntityType.FACT)[0]
    vector=agent.vectors.get(result.fact_id)
    cache=await agent.cache.get(domain='general',claim_text=request.claim_text)
    for provenance in [graph.properties['provenance'], vector.metadata['provenance'],json.loads(cache.evidence_summary)]:
        assert provenance['evidence']==evidence
        assert provenance['metadata']['execution_id']=='run-1'

    # Reopen persisted stores, not merely the same in-memory objects.
    from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
    from agents.memory_agent.vector_store.faiss_store import VectorStore
    from agents.memory_agent.cache.verification_cache import VerificationCache
    reopened_graph=KnowledgeGraph(persistence_path=agent._settings.kg_persistence_path)
    reopened_vectors=VectorStore(store_path=agent._settings.vector_store_path,
                                 embedding_model='all-MiniLM-L6-v2')
    reopened_vectors.initialize()
    reopened_cache=VerificationCache(db_path=agent._settings.cache_db_path,ttl=3600)
    await reopened_cache.initialize()
    try:
        persisted=[reopened_graph.find_entity_by_name(result.fact_id,EntityType.FACT)[0].properties['provenance'],
                   reopened_vectors.get(result.fact_id).metadata['provenance'],
                   json.loads((await reopened_cache.get(domain='general',claim_text=request.claim_text)).evidence_summary)]
        assert all(p['evidence']==evidence and p['metadata']['execution_id']=='run-1' for p in persisted)
    finally:
        await reopened_cache.close()

@pytest.mark.asyncio
async def test_vector_persistence_failure_is_not_reported_stored(agent,monkeypatch):
    def fail(): raise OSError('simulated isolated storage failure')
    monkeypatch.setattr(agent.vectors,'save',fail)
    result=await agent.store_facts_batch([StoreFactRequest(claim_text='A is located in B.',domain='general',verdict='verified',confidence=.9)])
    assert result.stored==0 and result.failed==1 and result.errors
