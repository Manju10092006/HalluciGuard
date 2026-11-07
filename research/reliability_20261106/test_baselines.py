import pytest
from baselines import Backend, Budget, Case, run

@pytest.mark.asyncio
async def test_factored_checks_do_not_receive_candidate_or_other_answers():
    prompts=[]
    outputs=iter(['["Who founded X?", "When was X founded?"]','Person A','1975','Final'])
    async def generate(prompt,temp,max_tokens):
        prompts.append(prompt)
        return {'status':'success','text':next(outputs)}
    result=await run('cove_style_factored',Case('x','Tell me about X','SECRET_DRAFT'),Backend(generate))
    assert result['status']=='produced' and result['calls']==4
    assert 'SECRET_DRAFT' not in prompts[1] and 'SECRET_DRAFT' not in prompts[2]
    assert 'Person A' not in prompts[2]

@pytest.mark.asyncio
async def test_budget_exhaustion_is_failure_not_release():
    async def generate(*args):return {'status':'success','text':'["A?"]'}
    result=await run('cove_style_factored',Case('x','Q','A'),Backend(generate,budget=Budget(max_calls=1)))
    assert result['status']=='failed' and result['answer']=='' and result['calls']==1

@pytest.mark.asyncio
async def test_rag_gate_withholds_missing_reports():
    async def generate(*args):return {'status':'success','text':'A.'}
    async def verify(*args):return []
    result=await run('rag_claim_verification',Case('x','Q',''),Backend(generate,verify=verify))
    assert result['status']=='withheld'

@pytest.mark.asyncio
async def test_selfcheck_does_not_invent_release_threshold():
    async def generate(*args):return {'status':'success','text':'A.'}
    async def nli(*args):return [{'contradiction_score':0.01}]*3
    result=await run('selfcheck_style_nli',Case('x','Q','A.'),Backend(generate,nli=nli))
    assert result['status']=='scored_only' and result['human_factuality'] is None
