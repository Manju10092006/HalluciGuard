"""Regression contracts derived from archived failures, without neural calls."""
import copy
import pytest
from api.pipeline import VerificationPipeline
from schemas.models import Passage
from scorers.evidence_scorer import EvidenceScorer

def passage(text):
    return Passage(title='',snippet=text,source='fixed_reference',source_id='fixed_reference',
                   url='https://reference.example/card',publication_date='',relevance_score=1)

def score(claim, text, raw, policy):
    scorer=EvidenceScorer(relation_policy=policy)
    p,n=VerificationPipeline._select_decision_grade_evidence([passage(text)],[copy.deepcopy(raw)],claim,scorer.relation_verifier)
    return scorer.score_evidence(claim,p,n,'general')['verdict'].value

NEUTRAL={'label':'neutral','entailment_score':.018024,'contradiction_score':.00044,'neutral_score':.981536}

def test_archived_location_failure_and_evidence_conditioned_repair():
    c='Veylan lies in North Merith.'
    assert score(c,c,NEUTRAL,'structured')=='unverified'
    assert score(c,c,NEUTRAL,'soft')=='verified'

@pytest.mark.parametrize('text',[
    'Veylan might lie in North Merith.', 'It is claimed that Veylan lies in North Merith.',
    'Veylan does not lie in North Merith.', 'Orlen lies in North Merith.',
    'Veylan lies in a different region.', '"Veylan lies in North Merith" is a false claim.',
])
def test_new_template_never_proves_a_missing_or_qualified_relation(text):
    assert score('Veylan lies in North Merith.',text,NEUTRAL,'soft')!='verified'

def test_unresolved_relation_keeps_raw_nli_without_forcing_support():
    c='Reid Hoffman is the primary founder of LinkedIn.'
    p=passage('The relevant biographical information is incomplete.')
    s=EvidenceScorer(relation_policy='soft')
    raw=copy.deepcopy(NEUTRAL)
    selected,_=VerificationPipeline._select_decision_grade_evidence([p],[raw],c,s.relation_verifier)
    assert selected==[] and raw==NEUTRAL

def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError): EvidenceScorer(relation_policy='typo')
