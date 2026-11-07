"""Replay frozen correction candidates with fresh CPU NLI and exact passages.

This isolates evidence selection and aggregation. It is not a hosted graph run.
"""
from common import *
configure()
import copy
import time

def main():
    import torch
    torch.set_num_threads(4)
    from api.pipeline import VerificationPipeline
    from nli import NLIEngine
    from schemas.models import Passage
    from scorers.evidence_scorer import EvidenceScorer
    from services.claim_analyzer import fallback_analyze
    from orchestration.graph import _answer_addresses_query
    cards=lines(TECH/'closed_reference_32.jsonl')
    inputs=[TECH/'closed_reference_32.jsonl',TECH/'reverification-probe/results.json']
    scorer=EvidenceScorer()
    engine=NLIEngine()
    rows=[]
    for card in [r for r in cards if r['scenario']=='mixed']:
        for label,candidate in [('valid',card['canonical']+' '+card['preserve']),
                                ('residual_false',card['false_claim']+' '+card['preserve']),
                                ('new_false',card['canonical']+' '+card['city']+' does not lie in '+card['region']+'.')]:
            start=time.perf_counter()
            analysis=fallback_analyze(candidate,card['query'],'general')
            reports=[]
            for c in analysis.factual_claims:
                passage=Passage(title=card['source_group']+' reference card',snippet=card['reference'],
                    source_id='fixed_reference',source='fixed_reference',url='https://reference.example/'+card['source_group'].lower(),
                    publication_date='',relevance_score=1,source_confidence_hint=1)
                raw=engine.batch_classify(c.claim_text,[passage.snippet])
                if any(r.get('degraded') for r in raw):
                    raise RuntimeError('Model inference degraded; no result may be reported')
                rel=scorer.relation_verifier.verify_relation(c.claim_text,[passage])
                passages,nli=VerificationPipeline._select_decision_grade_evidence([passage],copy.deepcopy(raw),c.claim_text,scorer.relation_verifier)
                score=scorer.score_evidence(c.claim_text,passages,nli,'general')
                reports.append({'claim':c.claim_text,'raw_nli':raw,'relation':rel.model_dump(mode='json'),
                                'selected':len(passages),'score':score,'passage':passage.model_dump(mode='json')})
            row={'id':card['id']+'_'+label,'label':label,'query':card['query'],'original':card['draft'],
                 'candidate':candidate,'reports':reports,'topical':_answer_addresses_query(card['query'],candidate),
                 'passed':bool(reports) and all(str(r['score']['verdict'].value)=='verified' for r in reports) and _answer_addresses_query(card['query'],candidate),
                 'seconds':time.perf_counter()-start}
            rows.append(row)
            print(row['id'],[(r['claim'],r['raw_nli'][0].get('label'),r['score']['verdict']) for r in reports],flush=True)
    save(RUN/'repair_diagnosis/predictions.json',rows)
    save(RUN/'repair_diagnosis/configuration.json',provenance(__file__,inputs,kind='fixed_passage_fresh_cpu_nli',models=engine.diagnostics(),n=len(rows),api_calls=0))

if __name__=='__main__': main()
