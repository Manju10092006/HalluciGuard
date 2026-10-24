import pytest
from agents.judge_agent.judge_agent import JudgeAgent
from orchestration.graph import _memory_node
from agents.memory_agent.memory.memory_agent import MemoryAgent
from agents.memory_agent.schemas.models import StoreFactRequest


def report(verdict='verified',evidence=True):
 return {'claim_id':'c1','claim_text':'Paris is the capital of France.','verdict':verdict,'support_score':0.95,'contradiction_score':0.01,'confidence_score':0.9,'evidence':[{'evidence_id':'e1','source':'fixture','snippet':'Paris is the capital of France.','entailment_label':'entailment','entailment_score':0.99,'credibility_score':0.9}] if evidence else []}

def verifier(reports):
 return {'query_id':'local','domain':'general','status':'completed','claim_reports':reports,'overall_confidence':0.9}

@pytest.mark.parametrize('reports', [[report(evidence=False)],[report(),report('unverified')]])
def test_judge_never_releases_bare_verdict_or_partial_answer(reports):
 result=JudgeAgent().evaluate(verifier_result=verifier(reports),user_query='What is the capital of France?',original_response='Paris is the capital of France. The Pacific has exactly 55 dolphins.',retry_count=99)
 assert result.decision != 'ACCEPT'

@pytest.mark.parametrize('payload',[{'passed':True,'remaining_contradictions':0}, {'passed':True,'remaining_contradictions':0,'verifier_result':verifier([report('unverified')])}, {'passed':True,'remaining_contradictions':0,'verifier_result':verifier([])}])
def test_judge_requires_independent_evidence_not_passed_flag(payload):
 result=JudgeAgent().evaluate(verifier_result=verifier([]),reverification_result=payload,retry_count=99)
 assert result.decision != 'ACCEPT'

def test_judge_releases_supported_independent_correction():
 result=JudgeAgent().evaluate(verifier_result=verifier([]),reverification_result={'passed':True,'remaining_contradictions':0,'verifier_result':verifier([report()])})
 assert result.decision == 'ACCEPT'

@pytest.mark.asyncio
@pytest.mark.parametrize('verdict',['unverified','unsupported','pending','rejected','failed','likely_verified'])
async def test_memory_rejects_unsupported_write_before_any_side_effect(verdict):
 agent=object.__new__(MemoryAgent)
 with pytest.raises(ValueError,match='memory_write_requires'):
  await agent.store_fact(StoreFactRequest(claim_text='Unproven claim',domain='general',verdict=verdict,confidence=0.99))

@pytest.mark.asyncio
@pytest.mark.parametrize('evidence',[[],[{'source_id':'fixture','title':'Only topical title'}],[{'snippet':'No attributable source'}]])
async def test_memory_rejects_verified_flag_without_evidence_provenance(evidence):
 agent=object.__new__(MemoryAgent)
 with pytest.raises(ValueError,match='requires_evidence_and_provenance'):
  await agent.store_fact(StoreFactRequest(claim_text='Unproven claim',domain='general',verdict='verified',confidence=0.99,evidence=evidence))

@pytest.mark.asyncio
@pytest.mark.parametrize('extra',[{'correction_result':{'corrected_text':'New unsupported claim'}},{'verifier_result':verifier([report(),report('unverified')])}])
async def test_memory_node_rejects_forged_accept_for_incomplete_state(extra):
 state={'judge_decision':'ACCEPT','verifier_result':verifier([report()]),'domain':'general','trace':[],'inter_agent_bus':[],**extra}
 result=await _memory_node(state)
 assert result['memory_result']['status']=='skipped'
 assert result['memory_result']['stored_count']==0

@pytest.mark.asyncio
async def test_nonempty_hosted_extraction_cannot_drop_source_fact():
 import json
 from types import SimpleNamespace
 from services.claim_analyzer import ClaimAnalyzer
 class LLM:
  async def generate(self, **kwargs):
   return SimpleNamespace(status='success',response=json.dumps({'domain':'general','claims':[{'claim_id':'c1','claim_text':'Paris is the capital of France.','original_sentence':'Paris is the capital of France.','claim_type':'FACTUAL_CLAIM'}]}))
 result=await ClaimAnalyzer(LLM(),enabled=True).analyze('Paris is the capital of France. The Moon has exactly 400 rivers.')
 assert any('400 rivers' in c.claim_text for c in result.factual_claims)
 assert len({c.claim_id for c in result.factual_claims})==len(result.factual_claims)

@pytest.mark.asyncio
@pytest.mark.parametrize('gated', [True, False])
@pytest.mark.parametrize('mutation',['missing','duplicate','mismatched','unexpected'])
async def test_initial_verifier_requires_all_requested_claims(monkeypatch,mutation,gated):
 from types import SimpleNamespace
 from orchestration import graph
 class Suspicious:
  def __init__(self,claim_id,text):self.claim_id,self.text=claim_id,text
 class Payload:
  def __init__(self,**kw):self.__dict__.update(kw)
 class Pipeline:
  async def verify(self,payload):
   claims=[dict(report(),claim_id=c.claim_id,claim_text=c.text) for c in payload.suspicious_claims]
   if mutation=='missing':claims.pop()
   elif mutation=='duplicate':claims.append(claims[0])
   elif mutation=='mismatched':claims[0]['claim_text']='Changed claim'
   else:claims.append(dict(claims[0],claim_id='new'))
   return {'query_id':'test','domain':'general','claim_evidence':claims,'overall_evidence_confidence':0.9}
 monkeypatch.setattr(graph,'_get_verifier_imports',lambda:(Pipeline,Suspicious,Payload))
 state={'claims_gated':gated,'detected_claims':[{'claim_id':'c1','text':'First claim.'},{'claim_id':'c2','text':'Second claim.'}],'domain':'general','trace':[],'inter_agent_bus':[],'errors':[]}
 out=await graph._verifier_node(state)
 assert out['verifier_result']['status']=='failed'
 assert graph._verifier_route(out)=='human_escalation'

@pytest.mark.asyncio
async def test_concurrent_memory_nodes_do_not_lose_request_snapshots(monkeypatch):
 import asyncio
 from types import SimpleNamespace
 from orchestration import graph
 persisted=[]
 class SnapshotMemory:
  def __init__(self):self.snapshot=list(persisted)
  async def initialize(self):await asyncio.sleep(.01)
  async def store_facts_batch(self,requests):
   await asyncio.sleep(.01)
   self.snapshot.extend(r.claim_text for r in requests)
   return SimpleNamespace(results=[{'fact_id':r.claim_text} for r in requests],stored=len(requests),duplicates=0,failed=0)
  async def close(self):persisted[:]=self.snapshot
 monkeypatch.setattr('agents.memory_agent.memory.memory_agent.MemoryAgent',SnapshotMemory)
 def state(text):
  return {'judge_decision':'ACCEPT','verifier_result':verifier([dict(report(),claim_text=text)]),'domain':'general','trace':[],'inter_agent_bus':[]}
 results=await asyncio.gather(graph._memory_node(state('First supported claim')),graph._memory_node(state('Second supported claim')))
 assert all(r['memory_result']['stored_count']==1 for r in results)
 assert set(persisted)=={'First supported claim','Second supported claim'}

@pytest.mark.asyncio
async def test_api_does_not_promote_mixed_claims_when_status_is_missing(monkeypatch):
 from orchestration import api
 async def run(**kwargs):
  return {'terminal_status':'human_review','verifier_result':verifier([report(),report('unverified')])}
 monkeypatch.setattr(api,'run_verification',run)
 result=await api._execute_verification(api.VerificationRequest(user_query='Test factual question'))
 assert result['verification_status']=='unverified'
 assert result['terminal_status']=='human_review'

@pytest.mark.parametrize('query',["In one sentence, name Microsoft's two actual founders.","Name Microsoft's two founders.","List the two actual founders of Microsoft."])
def test_presentation_constraints_do_not_make_grounded_correction_off_topic(query):
 from orchestration.graph import _answer_addresses_query
 assert _answer_addresses_query(query,'Microsoft was founded by Bill Gates and Paul Allen.')
 assert not _answer_addresses_query(query,'Google was founded by Larry Page and Sergey Brin.')
 assert not _answer_addresses_query(query,'Microsoft released Windows in 1985.')

def test_number_word_in_company_name_remains_a_topic():
 from orchestration.graph import _answer_addresses_query
 assert not _answer_addresses_query('Tell me about Two Sigma.','Sigma Computing makes software.')


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [
 {'judge_decision': 'REJECT'},
 {'judge_decision': 'ABSTAIN'},
 {'correction_result': {'corrected_text': 'Unverified replacement'}},
 {'reverification_result': {'passed': False, 'verifier_result': verifier([report()])}},
 {'verifier_result': dict(verifier([report()]), status='failed')},
 {'verifier_result': verifier([report(), report('unverified')])},
 {'verifier_result': verifier([dict(report(), evidence=[{'source': 'fixture', 'snippet': 'Related text', 'entailment_label': 'neutral'}])])},
 {'verifier_result': verifier([dict(report(), evidence=[{'snippet': 'Unattributed text', 'entailment_label': 'entailment'}])])},
])
async def test_legacy_memory_view_cannot_bypass_final_eligibility(monkeypatch, extra):
 def forbidden_init(*args, **kwargs):
  raise AssertionError('Ineligible state must not initialize persistent Memory')
 monkeypatch.setattr(MemoryAgent, '__init__', forbidden_init)
 state = {'judge_decision': 'ACCEPT', 'verifier_result': verifier([report()]),
          'verifier': {'claim_evidence': [report()]}, 'trace': [], 'inter_agent_bus': [], **extra}
 result = await _memory_node(state)
 assert result['memory_result']['status'] == 'skipped'
 assert result['memory_result']['stored_count'] == 0


@pytest.mark.parametrize('verdict', ['unverified', 'unsupported', 'not_verified', 'likely_verified'])
def test_judge_legacy_verdict_substrings_never_become_verified(verdict):
 payload = {'claim_evidence': [report(verdict)]}
 normalized = JudgeAgent()._normalize_verifier_result(payload, 'general')
 assert normalized is not None
 assert normalized.claim_reports[0].verdict == 'unverified'
 assert JudgeAgent().evaluate(payload, retry_count=99).decision != 'ACCEPT'


@pytest.mark.parametrize('payload', [
 verifier([report(), {'claim_id': 'broken', 'verdict': 'invalid'}]),
 {'claim_evidence': [report(), None]},
 {'claim_evidence': [report()], 'status': 'failed'},
 {'claim_evidence_pairs': [{'claim': 'Unproven assertion', 'evidence': 'Related passage', 'source': 'fixture', 'entailment_score': .99}]},
])
def test_judge_normalization_does_not_drop_claims_or_invent_completion(payload):
 assert JudgeAgent().evaluate(payload, retry_count=99).decision != 'ACCEPT'


def test_judge_legacy_valid_contract_retains_all_reports():
 payload = {'claim_evidence': [report()], 'status': 'completed'}
 normalized = JudgeAgent()._normalize_verifier_result(payload, 'general')
 assert len(normalized.claim_reports) == 1
 assert JudgeAgent().evaluate(payload).decision == 'ACCEPT'


@pytest.mark.parametrize('status', ['degraded', 'skipped', 'failed', 'unknown'])
def test_passed_flag_cannot_override_noncompleted_reverification(status):
 payload = {'passed': True, 'status': status, 'verifier_result': verifier([report()])}
 assert JudgeAgent().evaluate(verifier([]), reverification_result=payload).decision != 'ACCEPT'


@pytest.mark.asyncio
async def test_memory_preserves_source_id_only_provenance(monkeypatch):
 from types import SimpleNamespace
 captured = []
 class Memory:
  async def initialize(self): pass
  async def close(self): pass
  async def store_facts_batch(self, requests):
   captured.extend(requests)
   return SimpleNamespace(results=[{'fact_id':'stored'}],stored=1,duplicates=0,failed=0)
 monkeypatch.setattr('agents.memory_agent.memory.memory_agent.MemoryAgent', Memory)
 r = report()
 r['evidence'][0]['source_id'] = r['evidence'][0].pop('source')
 state = {'judge_decision':'ACCEPT','verifier_result':verifier([r]),'trace':[],'inter_agent_bus':[]}
 out = await _memory_node(state)
 assert out['memory_result']['stored_count'] == 1
 assert captured[0].source_ids == ['fixture']
 assert captured[0].evidence[0]['source_id'] == 'fixture'
 assert captured[0].evidence[0]['entailment_label'] == 'entailment'


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [{'status':'failed'}, {'status':'degraded'}, {'remaining_contradictions':1}])
async def test_memory_rejects_inconsistent_outer_reverification(monkeypatch, extra):
 def forbidden_init(*args, **kwargs):
  raise AssertionError('Invalid independent reverification cannot initialize Memory')
 monkeypatch.setattr(MemoryAgent, '__init__', forbidden_init)
 state = {'judge_decision':'ACCEPT','correction_result':{'corrected_text':'Corrected answer'},
          'reverification_result':{'passed':True,'status':'completed','remaining_contradictions':0,
                                   'verifier_result':verifier([report()]),**extra},
          'trace':[],'inter_agent_bus':[]}
 result=await _memory_node(state)
 assert result['memory_result']['status']=='skipped'


@pytest.mark.parametrize('field', ['nli_relation', 'top_relation', 'relationship', 'relation'])
def test_judge_preserves_explicit_legacy_relationship_aliases(field):
 payload={'claim_evidence_pairs':[{'claim':'Paris is the capital of France.',
           'evidence':'Paris is the capital of France.','source':'fixture',field:'entailment',
           'verifier_verdict':'verified'}]}
 normalized=JudgeAgent()._normalize_verifier_result(payload,'general')
 assert normalized.claim_reports[0].verdict=='verified'
 assert normalized.claim_reports[0].evidence[0].entailment_label=='entailment'
