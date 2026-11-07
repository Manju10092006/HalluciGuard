import pytest
from orchestration import graph
from agents.memory_agent.memory.memory_agent import MemoryAgent

@pytest.mark.asyncio
async def test_initialization_failure_has_canonical_result_without_revoking_release(monkeypatch):
    async def fail(self): raise OSError('isolated initialization failure')
    monkeypatch.setattr(MemoryAgent,'initialize',fail)
    state={'judge_decision':'ACCEPT','final_response':'Verified answer.', 'trace':[],
           'verifier_result':{'claim_reports':[{'claim_id':'a','claim_text':'Verified answer.','verdict':'verified'}]}}
    result=await graph._memory_node(state)
    assert result['memory_result']['status']=='failed'
    assert result['terminal_status']=='accepted'
    assert result['final_response']=='Verified answer.'
