"""Frozen candidate evaluated once on separately sampled ANLI R2 test pairs."""
from common import *
configure()
from zipfile import ZipFile
import copy
import random
import time
from collections import Counter

def main():
    archive=ARCHIVE/'technical_20261105/external/anli_v1.0.zip'
    with ZipFile(archive) as z:
        raw=z.read('anli_v1.0/R2/test.jsonl')
    corpus=[json.loads(x) for x in raw.splitlines()]
    rng=random.Random(20261106)
    rows=[r for label in 'ecn' for r in rng.sample(sorted([r for r in corpus if r['label']==label],key=lambda r:r['uid']),40)]
    save(RUN/'heldout/selection.json',{'dataset':'ANLI v1.0 R2 test','seed':20261106,'n':len(rows),'ids':[r['uid'] for r in rows],
        'source_sha256':hashlib.sha256(raw).hexdigest(),'selection':'40 per reference class, before inference',
        'candidate':'soft, frozen after the 100-case development replay','prior_round':'R3 was used in earlier work'})
    save(RUN/'heldout/configuration.json',provenance(__file__,[archive],kind='fresh_cpu_inference_frozen_premises',api_calls=0))
    import torch
    torch.set_num_threads(4)
    from api.pipeline import VerificationPipeline
    from schemas.models import Passage
    from nli import NLIEngine
    from scorers.evidence_scorer import EvidenceScorer
    from claims.claim_merger import ClaimMerger
    engine=NLIEngine();scorers={m:EvidenceScorer(relation_policy=m) for m in ['structured','no_scoped','soft','nli_temporal']}
    mapping={'e':'verified','c':'contradicted','n':'unverified'}
    predictions=[]
    for i,row in enumerate(rows):
        passage=Passage(title='',snippet=row['context'],source='fixed_reference',source_id='fixed_reference',
                        url='https://reference.example/anli-r2',publication_date='',relevance_score=1)
        start=time.perf_counter();raw=engine.batch_classify(row['hypothesis'],[row['context']])
        if any(x.get('degraded') for x in raw): raise RuntimeError('Degraded NLI is not a valid experiment')
        item={'id':row['uid'],'premise':row['context'],'claim':row['hypothesis'],'gold':mapping[row['label']],
              'raw_nli':raw,'inference_seconds':time.perf_counter()-start,'policies':{}}
        for mode,scorer in scorers.items():
            p,n=VerificationPipeline._select_decision_grade_evidence([passage],copy.deepcopy(raw),row['hypothesis'],scorer.relation_verifier)
            score=scorer.score_evidence(row['hypothesis'],p,n,'general')
            item['policies'][mode]=ClaimMerger().merge_results([{'scores':score,'evidence_items':[]}])['verdict']
        predictions.append(item)
        if (i+1)%20==0: print('ANLI R2',i+1,flush=True)
    summary={m:{'n':len(predictions),'agreement':sum(r['policies'][m]==r['gold'] for r in predictions),
                'confusion':dict(Counter(r['gold']+' -> '+r['policies'][m] for r in predictions)),
                'false_support':sum(r['gold']!='verified' and r['policies'][m]=='verified' for r in predictions),
                'false_contradiction':sum(r['gold']!='contradicted' and r['policies'][m]=='contradicted' for r in predictions)} for m in scorers}
    save(RUN/'heldout/predictions.json',predictions);save(RUN/'heldout/summary.json',summary)
    print(summary)

if __name__=='__main__':main()
