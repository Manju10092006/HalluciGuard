"""Matched archived Judge states and verifier pairs, with no provider calls."""
from common import *
configure()
import copy
from collections import Counter
import time

def main():
    import numpy as np
    from agents.judge_agent.config import JudgeConfig
    from agents.judge_agent.judge_agent import JudgeAgent
    from api.pipeline import VerificationPipeline
    from scorers.evidence_scorer import EvidenceScorer
    from claims.claim_merger import ClaimMerger
    from schemas.models import Passage
    states=lines(PRIOR/'live_challenge_analysis/judge_policy_inputs.jsonl')
    history={r['id']:r for r in load(PRIOR/'control_replay/release_policy_cases.json')}
    rows=[]
    for item in states:
        state=item['input_state']
        args=dict(verifier_result=state.get('verifier_result') or state.get('verifier',{}),
            detector_result=state.get('detector_result') or state.get('detector',{}),
            user_query=state['user_query'],original_response=state.get('llm_response') or state.get('draft_response'),
            domain=state.get('domain','general'),reverification_result=state.get('reverification_result'),
            retry_count=state.get('retry_count',0),correction_attempt_count=state.get('correction_attempt_count',0))
        row={'case_id':item['case_id'],'input':args,'recorded':item['recorded_judge_result'],'policies':{}}
        for mode in ['default','strict_all_verified','slot_aware']:
            start=time.perf_counter()
            result=JudgeAgent(JudgeConfig(release_policy=mode)).evaluate(**copy.deepcopy(args)).model_dump(mode='json')
            row['policies'][mode]={'result':result,'seconds':time.perf_counter()-start}
        no_detector=copy.deepcopy(args);no_detector['detector_result']=None
        row['no_detector_decision']=JudgeAgent().evaluate(**no_detector).decision
        rows.append(row)
    last={r['case_id']:r for r in rows}
    summary={'states':len(rows),'last_states':len(last),'human_review':'pending',
             'detector_action_changes':sum(r['no_detector_decision']!=r['policies']['default']['result']['decision'] for r in rows),
             'historical_action_mismatches':sum(r['policies']['default']['result']['decision']!=r['recorded']['decision'] for r in rows),'policies':{}}
    for mode in ['default','strict_all_verified','slot_aware']:
        accepted=[r for r in last.values() if r['policies'][mode]['result']['decision']=='ACCEPT']
        summary['policies'][mode]={'accepted':len(accepted),
            'known_false_released':sum(history[r['case_id']]['known_false_relation_present'] is True for r in accepted),
            'target_pass_released':sum(history[r['case_id']]['target_relation_pass'] is True for r in accepted),
            'decisions_all_states':dict(Counter(r['policies'][mode]['result']['decision'] for r in rows)),
            'previously_accepted_withheld':[r['case_id'] for r in last.values() if history[r['case_id']]['default_release'] and r not in accepted]}
    save(RUN/'judge/predictions.json',rows)
    save(RUN/'judge/summary.json',summary)
    print('Judge',summary,flush=True)
    variants=['structured','no_scoped','soft','nli_temporal']
    scorers={m:EvidenceScorer(relation_policy=m) for m in variants}
    mapping={'SUPPORTED':'verified','CONTRADICTED':'contradicted','INSUFFICIENT_EVIDENCE':'unverified'}
    predictions=[];inputs=[]
    for path in sorted((PRIOR/'relation_replay').glob('*.json')):
        case=load(path)
        if not isinstance(case,dict) or not isinstance(case.get('reports'),list): continue
        inputs.append(path)
        for report in case['reports']:
            if report.get('error') or report.get('reference_label') is None: continue
            row={'case_id':case['id'],'source_group':case['source_group'],'claim':report['claim_text'],
                 'reference':mapping[report['reference_label']],'historical':{k:report[k]['verdict'] for k in ['guard_enabled','guard_disabled']},'policies':{}}
            for mode,scorer in scorers.items():
                components=[];diagnostics=[]
                for snapshot,batch in zip(report['snapshots'],report['fresh_nli_batches']):
                    claim=snapshot['claim_text'];passages=[Passage.model_validate(p) for p in batch['passages']]
                    raw=copy.deepcopy(batch['raw_nli'])
                    selected,scores=VerificationPipeline._select_decision_grade_evidence(passages,raw,claim,scorer.relation_verifier)
                    values=scorer.score_evidence(claim,selected,scores,'general')
                    components.append({'scores':values,'evidence_items':[]})
                    for p,nli in zip(passages,batch['raw_nli']):
                        rel=scorer.relation_verifier.verify_relation(claim,[p])
                        diagnostics.append({'claim':claim,'passage':p.model_dump(mode='json'),'nli':nli,
                            'relation':rel.model_dump(mode='json'),'selected':p in selected,
                            'temporal_excluded':scorer.relation_verifier.is_temporally_inapplicable(claim,p)})
                merged=ClaimMerger().merge_results(components)
                row['policies'][mode]={'verdict':getattr(merged.get('verdict'),'value',merged.get('verdict')) or 'unverified','details':diagnostics}
            predictions.append(row)
    summary={'n':len(predictions),'cohort':'previously inspected development archive, no held-out claim', 'policies':{}}
    for mode in variants:
        correct=[r for r in predictions if r['policies'][mode]['verdict']==r['reference']]
        summary['policies'][mode]={'agreement':len(correct),'confusion':dict(Counter(r['reference']+' -> '+r['policies'][mode]['verdict'] for r in predictions)),
          'false_support':sum(r['reference']!='verified' and r['policies'][mode]['verdict']=='verified' for r in predictions),
          'false_contradiction':sum(r['reference']!='contradicted' and r['policies'][mode]['verdict']=='contradicted' for r in predictions),
          'helped_vs_structured':sum(r['policies'][mode]['verdict']==r['reference'] and r['policies']['structured']['verdict']!=r['reference'] for r in predictions),
          'harmed_vs_structured':sum(r['policies'][mode]['verdict']!=r['reference'] and r['policies']['structured']['verdict']==r['reference'] for r in predictions)}
    save(RUN/'relation/predictions.json',predictions)
    save(RUN/'relation/summary.json',summary)
    save(RUN/'replay_configuration.json',provenance(__file__,inputs+[PRIOR/'live_challenge_analysis/judge_policy_inputs.jsonl'],mode='frozen_scores_replay',api_calls=0))
    print('Relation',summary,flush=True)

if __name__=='__main__': main()
