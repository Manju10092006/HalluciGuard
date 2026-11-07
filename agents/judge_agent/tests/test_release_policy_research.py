import pytest
from agents.judge_agent.config import JudgeConfig
from agents.judge_agent.judge_agent import JudgeAgent
from agents.judge_agent.answer_slots import same_answer_slot

def decide(verified, unknown, policy='slot_aware', unresolved='unverified'):
    reports=[{'claim_id':'v','claim_text':verified,'verdict':'verified','support_score':.9,'confidence_score':.9},
             {'claim_id':'u','claim_text':unknown,'verdict':unresolved,'support_score':0,'confidence_score':0}]
    return JudgeAgent(JudgeConfig(release_policy=policy)).evaluate(
        verifier_result={'query_id':'fixture','domain':'general','status':'completed','claim_reports':reports},
        user_query=verified, original_response=verified+' '+unknown, retry_count=2).decision

@pytest.mark.parametrize('verified,unknown',[
    ('Paris is the capital of France.','London is the capital of France.'),
    ('James Gosling created Java.','Snehith Reddy created Java.'),
    ('Richard Hipp is SQLite\'s lead developer.','Snehith Reddy is SQLite\'s lead developer.'),
    ('Microsoft was founded by Bill Gates.','Microsoft was founded by Paul Allen.'),
    ('Nairobi is the capital of Kenya.','Nairobi is not the capital of Kenya.'),
    ('Nairobi is the capital of Kenya.','London might be the capital of Kenya.'),
])
def test_competing_unverified_answers_are_withheld_without_declaring_them_false(verified,unknown):
    assert decide(verified,unknown)=='ABSTAIN'
    assert decide(verified,unknown,'default')=='ACCEPT'

def test_distinct_slot_elaboration_retains_historical_coverage():
    assert decide('Microsoft was founded by Bill Gates.','Microsoft is based in Redmond.')=='ACCEPT'
    assert decide('Microsoft was founded by Bill Gates.','Microsoft is based in Redmond.','strict_all_verified')=='ABSTAIN'

def test_temporal_scopes_are_distinct_and_aliases_are_explicit():
    assert not same_answer_slot('Bonn was the capital of Germany in 1980.',['Berlin is the capital of Germany.'])
    assert same_answer_slot('London is the capital of the Republic of Kenya.',['Nairobi is the capital of Kenya.'],{'republic of kenya':'kenya'})
    assert not same_answer_slot('Paris is the capital of Indiana.',['Paris is the capital of India.'])

def test_conflicting_evidence_cannot_be_tolerated_in_slot_policy():
    assert decide('Microsoft was founded by Bill Gates.','The sea is deep.',unresolved='conflicted')!='ACCEPT'

def test_multiple_verified_cofounders_are_accepted():
    reports=[{'claim_id':str(i),'claim_text':text,'verdict':'verified','support_score':.9}
             for i,text in enumerate(['Microsoft was founded by Bill Gates.','Microsoft was founded by Paul Allen.'])]
    result=JudgeAgent(JudgeConfig(release_policy='slot_aware')).evaluate(
        verifier_result={'query_id':'fixture','domain':'general','status':'completed','claim_reports':reports},
        user_query='Who founded Microsoft?')
    assert result.decision=='ACCEPT'

def test_policy_typo_is_not_silently_defaulted():
    with pytest.raises(ValueError): JudgeConfig(release_policy='typo')
