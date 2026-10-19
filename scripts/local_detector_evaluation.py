"""Fresh local checkpoint evaluation; source datasets/checkpoints are read-only."""
import argparse,json,os,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}.items():os.environ.setdefault(k,v)
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
import numpy as np
import torch
from halluciguard_detector.detector import Detector
from halluciguard_detector.training import predict_rows,evaluate_saved_predictions
root=Path(__file__).resolve().parents[1];out=root/'runtime-logs'/'fresh-model-evaluation';out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in (a.data_dir/'test.jsonl').read_text(encoding='utf-8').splitlines()]
seen=set()
for split in ['train','dev']:
 for line in (a.data_dir/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
  r=json.loads(line);seen.add((r['claim'].strip(),r['evidence'].strip()))
unique=set();eligible=[]
for i,r in enumerate(rows):
 key=(r['claim'].strip(),r['evidence'].strip())
 if key not in seen and key not in unique:eligible.append(i);unique.add(key)
rng=np.random.default_rng(20261019);chosen=[]
for label in [0,1,2]:
 candidates=[i for i in eligible if rows[i]['label_id']==label]
 chosen.extend(rng.choice(candidates,size=min(32,len(candidates)),replace=False).tolist())
chosen.sort()
indices=list(range(len(rows))) if a.full else chosen
model=Detector(root/'artifacts'/'detector-best')
print('device',str(model.device),'fresh_rows',len(indices),'runtime_rows',24,flush=True)
logits_parts=[];label_parts=[];started=time.monotonic()
for start in range(0,len(indices),128):
 selected=indices[start:start+128]
 logits,labels=predict_rows(model.model,model.tokenizer,[rows[i] for i in selected],device=model.device,max_length=model.max_length,batch_size=16)
 logits_parts.append(logits);label_parts.append(labels)
 print('fresh_pairs',start+len(selected),'seconds',round(time.monotonic()-started,2),flush=True)
logits=np.concatenate(logits_parts);labels=np.concatenate(label_parts)
metrics=evaluate_saved_predictions(logits,labels,model.temperature,model.contradiction_threshold,model.verification_risk_threshold)
stored=np.load(root/'artifacts'/'detector-best'/'test_predictions.npz')
assert np.array_equal(stored['labels'][indices],labels)
report={'full_historical_pairs':a.full,'seed':20261019,'sample_selection':('all historical test pairs; additional metrics exclude split overlap and duplicate pairs' if a.full else '32 per class, excluding any train/dev exact pair and within-test duplicate pair; no threshold fitting'),'rows':len(indices),'eligible_unique_test_pairs':len(eligible),'indices':indices,'device':str(model.device),'seconds':time.monotonic()-started,'max_abs_saved_logit_difference':float(np.max(np.abs(logits-stored['logits'][indices]))),'metrics':metrics}
if a.full:
 report['leakage_filtered_unique_test_metrics']=evaluate_saved_predictions(logits[eligible],labels[eligible],model.temperature,model.contradiction_threshold,model.verification_risk_threshold)
(out/('fresh-pairwise-full.json' if a.full else 'fresh-pairwise.json')).write_text(json.dumps(report,indent=2))
# Evaluate a fixed subset through actual production decomposition/selection/NLI.
runtime_indices=[]
for label in [0,1,2]:runtime_indices.extend([i for i in chosen if rows[i]['label_id']==label][:8])
runtime=[]
for i in runtime_indices:
 r=rows[i];res=model.detect(r['claim'],[r['evidence']])
 predicted=1 if res.contradicted_count else 2 if res.unknown_count or not res.claim_count else 0
 runtime.append({'index':i,'id':r['id'],'label_id':r['label_id'],'prediction':predicted,'response':res.model_dump(mode='json')})
 print('runtime_pairs',len(runtime),flush=True)
from sklearn.metrics import confusion_matrix,f1_score
runtime_report={'rows':len(runtime),'selection':'first 8 of each class from the fixed stratified leakage-filtered sample','macro_f1':float(f1_score([r['label_id'] for r in runtime],[r['prediction'] for r in runtime],average='macro')),'confusion_matrix':confusion_matrix([r['label_id'] for r in runtime],[r['prediction'] for r in runtime],labels=[0,1,2]).tolist(),'cases':runtime}
(out/'fresh-runtime.json').write_text(json.dumps(runtime_report,indent=2))
print('complete',{'pairwise_macro_f1':metrics['three_class']['macro_f1'],'saved_logit_max_difference':report['max_abs_saved_logit_difference'],'runtime_macro_f1':runtime_report['macro_f1']},flush=True)
