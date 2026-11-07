"""Derive uncertainty, error inventory and blinded annotation from raw records."""
from common import *
import csv
import math
from collections import Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def interval(delta, groups, seed=20261106):
    """Paired percentile bootstrap; resample source groups with all rows intact."""
    rng=np.random.default_rng(seed)
    names=sorted(set(groups));ix={g:[i for i,x in enumerate(groups) if x==g] for g in names}
    values=np.asarray(delta,dtype=float)
    estimates=[float(values[[i for g in rng.choice(names,len(names),replace=True) for i in ix[g]]].mean()) for _ in range(4000)]
    return {'difference':float(values.mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'groups':len(names),'resamples':4000,'seed':seed}


def csv_write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    relation=load(RUN/'relation/predictions.json');heldout=load(RUN/'heldout/predictions.json')
    report={};inventory=[]
    labels=['verified','contradicted','unverified','conflicted']
    for name,rows in [('development',relation),('heldout_r2',heldout)]:
        gold=[r.get('reference',r.get('gold')) for r in rows]
        pred=lambda r,m: r['policies'][m]['verdict'] if isinstance(r['policies'][m],dict) else r['policies'][m]
        base=np.array([pred(r,'structured')==g for r,g in zip(rows,gold)],int)
        report[name]={}
        for mode in ['structured','no_scoped','soft','nli_temporal']:
            output=[pred(r,mode) for r in rows]
            precision,recall,f1,support=precision_recall_fscore_support(gold,output,labels=labels,zero_division=0)
            correct=np.array([p==g for p,g in zip(output,gold)],int)
            report[name][mode]={'n':len(rows),'agreement':int(correct.sum()),
                'per_class':{label:{'precision':float(p),'recall':float(r),'f1':float(f),'n':int(s)} for label,p,r,f,s in zip(labels,precision,recall,f1,support)},
                'macro_f1_reference_classes':float(np.mean(f1[np.array(support)>0])),
                'paired_vs_structured':interval(correct-base,[r.get('source_group',r.get('id')) for r in rows]),
                'helped':int(((correct==1)&(base==0)).sum()),'harmed':int(((correct==0)&(base==1)).sum())}
    for row in relation:
        details=row['policies']['structured']['details']
        reasons=[]
        if any(d['temporal_excluded'] for d in details):reasons.append('temporal_screen')
        if details and not any(d['selected'] for d in details):reasons.append('all_decision_passages_excluded')
        if any(not d['relation'].get('claim_triple') for d in details):reasons.append('no_claim_triple')
        if any(d['relation'].get('status')=='NO_TRIPLE_EXTRACTED' for d in details):reasons.append('inconclusive_relation')
        if not reasons:reasons=['aggregation_or_pair_scoring_requires_review']
        inventory.append({**row,'automatic_diagnostic_tags':reasons,'human_cause_adjudication':None})
    save(RUN/'analysis/verifier_metrics.json',report)
    save(RUN/'analysis/relation_error_inventory.json',inventory)
    states=load(RUN/'judge/predictions.json');last={r['case_id']:r for r in states}
    references={r['id']:r for r in load(PRIOR/'control_replay/release_policy_cases.json')}
    judges={}
    for mode in ['default','strict_all_verified','slot_aware']:
        delta=[];false_delta=[];groups=[]
        for id,row in last.items():
            accepted=row['policies'][mode]['result']['decision']=='ACCEPT'
            original=row['policies']['default']['result']['decision']=='ACCEPT'
            delta.append(int(accepted)-int(original))
            false_delta.append((int(accepted)-int(original))*int(references[id]['known_false_relation_present'] is True))
            groups.append(id.rsplit('_',1)[0] if not id.endswith('insufficient_future') else id[:-20])
        # Use actual source group IDs from the original fixed challenge dataset.
        mapping={r['id']:r['source_group'] for r in lines(PRIOR/'datasets/challenge_100.jsonl')}
        groups=[mapping[id] for id in last]
        judges[mode]={'coverage_difference':interval(delta,groups),'known_false_release_difference':interval(false_delta,groups),
                      'whole_answer_correct_releases':None,'whole_answer_correct_withheld':None,'human_review_outcomes':None}
    save(RUN/'analysis/judge_intervals.json',judges)
    annotations=[]
    def add(cohort,id,group,q,answer,claim,evidence,original='',kind=''):
        annotations.append({'item_id':f'{cohort}:{id}:{len(annotations)+1}','cohort':cohort,'group':group,
            'question':q,'answer':answer,'original_answer':original,'atomic_claim':claim,
            'evidence_json':json.dumps(evidence,ensure_ascii=True),'error_origin':kind,
            'support_label':'','importance':'','correction_validity':'','correct_answer_withheld':'',
            'new_unsupported_addition':'','annotator_id':'','notes':''})
    for id,row in last.items():
        inp=row['input'];reports=inp['verifier_result'].get('claim_reports',[])
        for c in reports:
            add('challenge',id,mapping[id],inp['user_query'],inp['original_response'],c['claim_text'],c.get('evidence',[]),kind='authored_perturbation_or_supported_control')
    for row in heldout:
        add('anli_r2',row['id'],row['id'],'Assess the hypothesis against the premise.',row['claim'],row['claim'],[{'snippet':row['premise']}],kind='established_adversarial_nli')
    for row in load(RUN/'repair_diagnosis/predictions.json'):
        for c in row['reports']:
            add('repair',row['id'],row['id'].split('_')[0],row['query'],row['candidate'],c['claim'],[c['passage']],row['original'],'fictional_fixed_card')
    # Newly generated whole answers remain blank for human segmentation and review.
    for path in sorted((RUN/'external_baselines').glob('baseline_*.json')):
        row=load(path)
        if row.get('answer'):
            original=next(r for r in load(PRIOR/'runtime_records/paired_baselines/results.json') if r['id']==row['id'])
            add('baseline_pilot',row['id']+':'+row['method'],row['source_group'],original['query'],row['answer'],'',original['shared_first_pass_evidence'],original['initial_draft'],'new_model_output')
    for annotator in ['annotator_a','annotator_b']:
        csv_write(RUN/f'annotation/{annotator}.csv',annotations,list(annotations[0]))
    save(RUN/'annotation/manifest.json',{'rows':len(annotations),'answer_groups':len(set((r['cohort'],r['answer'],r['question']) for r in annotations)),
        'human_completed':0,'labels':'blank','model_verdicts_hidden':True,
        'note':'Claim lists from system extraction or fixed examples, not a completeness guarantee. Baseline pilot claims require independent segmentation.'})
    save(RUN/'analysis/configuration.json',provenance(__file__,[RUN/'relation/predictions.json',RUN/'heldout/predictions.json',RUN/'judge/predictions.json'],kind='deterministic_statistics_and_annotation_preparation'))
    print('Derived statistics and',len(annotations),'blank annotation rows.')

if __name__=='__main__':main()
