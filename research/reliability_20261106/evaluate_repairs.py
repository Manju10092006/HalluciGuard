"""Execute the real ReVerifier and Judge gates on frozen neural pair scores."""
from common import *
configure()
import asyncio
import copy
import time
from unittest.mock import patch

async def main():
    from orchestration import graph
    from services.claim_analyzer import ClaimAnalyzer, fallback_analyze
    from schemas.models import SuspiciousClaim,VerifierInputV2,Passage
    from api.pipeline import VerificationPipeline
    from scorers.evidence_scorer import EvidenceScorer
    from agents.judge_agent.judge_agent import JudgeAgent
    from agents.judge_agent.config import JudgeConfig
    rows=load(RUN/'repair_diagnosis/predictions.json')
    outputs=[]
    for mode in ['structured','soft']:
        scorer=EvidenceScorer(relation_policy=mode)
        for row in rows:
            class FrozenPipeline:
                async def verify(self,payload):
                    reports=[]
                    for claim in payload.suspicious_claims:
                        reference=next(r for r in row['reports'] if r['claim']==claim.text)
                        passages,nli=VerificationPipeline._select_decision_grade_evidence(
                            [Passage.model_validate(reference['passage'])],copy.deepcopy(reference['raw_nli']),claim.text,scorer.relation_verifier)
                        score=scorer.score_evidence(claim.text,passages,nli,'general')
                        reports.append({'claim_id':claim.claim_id,'claim_text':claim.text,
                                        **score,'evidence':[p.model_dump(mode='json') for p in passages]})
                    return {'query_id':payload.query_id,'domain':'general','claim_evidence':reports}
            async def analyze(self,candidate,query,domain):
                return fallback_analyze(candidate,query,domain)
            state={'request_id':row['id'],'user_query':row['query'],'domain':'general',
                   'llm_response':row['original'],'correction_result':{'corrected_text':row['candidate']},
                   'correction_attempt_count':1,'trace':[]}
            start=time.perf_counter()
            with patch.object(ClaimAnalyzer,'analyze',analyze),patch.object(graph,'_get_verifier_imports',lambda:(FrozenPipeline,SuspiciousClaim,VerifierInputV2)):
                update=await graph._reverifier_node(state)
            result=update['reverification_result']
            decision=JudgeAgent(JudgeConfig(release_policy='slot_aware' if mode=='soft' else 'default')).evaluate(
                verifier_result=result['verifier_result'],reverification_result=result,
                user_query=row['query'],original_response=row['candidate'],correction_attempt_count=1)
            outputs.append({'id':row['id'],'label':row['label'],'mode':mode,'state':state,'result':result,
                            'judge':decision.model_dump(mode='json'),'seconds':time.perf_counter()-start})
    summary={}
    for mode in ['structured','soft']:
        subset=[r for r in outputs if r['mode']==mode]
        valid=[r for r in subset if r['label']=='valid'];invalid=[r for r in subset if r['label']!='valid']
        summary[mode]={'valid_n':len(valid),'valid_accepted':sum(r['judge']['decision']=='ACCEPT' for r in valid),
          'invalid_n':len(invalid),'invalid_withheld':sum(r['judge']['decision']!='ACCEPT' for r in invalid),
          'false_acceptance':sum(r['judge']['decision']=='ACCEPT' for r in invalid),
          'false_rejection':sum(r['judge']['decision']!='ACCEPT' for r in valid),
          'complete_report_coverage':sum(r['result']['claim_coverage']['complete'] for r in subset),
          'mean_gate_seconds':sum(r['seconds'] for r in subset)/len(subset)}
    save(RUN/'repairs/predictions.json',outputs)
    save(RUN/'repairs/summary.json',summary)
    save(RUN/'repairs/configuration.json',provenance(__file__,[RUN/'repair_diagnosis/predictions.json'],
         kind='real_ReVerifier_and_Judge_over_frozen_NLI',api_calls=0,
         limitation='Known diagnostic cards; same archived groups, not held-out factuality. Extraction uses source-preserving fallback.'))
    print(summary)

if __name__=='__main__':asyncio.run(main())
