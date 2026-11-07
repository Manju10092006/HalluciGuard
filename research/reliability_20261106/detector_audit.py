"""Recompute complete classifier metrics and data overlap without retraining."""
from common import *
configure()
from collections import Counter,defaultdict
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,confusion_matrix

def metrics(prob,y,threshold):
    risk=1-prob[:,0];binary=y!=0;pred=risk>=threshold;classes=prob.argmax(1)
    tn,fp,fn,tp=confusion_matrix(binary,pred,labels=[False,True]).ravel()
    bins=np.minimum((risk*10).astype(int),9)
    return {'n':len(y),'class_counts':dict(Counter(map(int,y))), 'threshold':threshold,
      'binary_accuracy':accuracy_score(binary,pred),'macro_f1_3class':f1_score(y,classes,average='macro'),
      'binary_f1':f1_score(binary,pred,zero_division=0),'precision':precision_score(binary,pred,zero_division=0),
      'recall':recall_score(binary,pred,zero_division=0),'contradiction_triage_recall':float(pred[y==1].mean()),
      'false_positive_rate':fp/(fp+tn),'false_negative_rate':fn/(fn+tp),
      'roc_auc':roc_auc_score(binary,risk),'average_precision':average_precision_score(binary,risk),
      'ece10':sum(float((bins==i).mean())*abs(float(risk[bins==i].mean())-float(binary[bins==i].mean())) for i in range(10) if (bins==i).any()),
      'binary_confusion':[[int(tn),int(fp)],[int(fn),int(tp)]],
      'multiclass_confusion':confusion_matrix(y,classes,labels=[0,1,2]).tolist(),
      'flagged_pairs':int(pred.sum()),'verification_without_bypass':len(y)}

def main():
    checkpoint=ROOT/'artifacts/detector-best';cal=load(checkpoint/'calibration.json')
    a=np.load(checkpoint/'test_predictions.npz')
    historical=metrics(softmax(a['logits']/cal['temperature'],axis=1),a['labels'],cal['hallucination_threshold'])
    reports={'historical_logits':historical}
    profile=load(TECH/'detector/detector_profile.json')
    inputs=[checkpoint/'test_predictions.npz',checkpoint/'calibration.json']
    for cohort in ['reserved_ragtruth','anli_round3']:
        p=TECH/f'detector/{cohort}_predictions.npz';inputs.append(p);a=np.load(p)
        original=softmax(a['logits_lexical_256']/cal['temperature'],axis=1)
        original[a['guard'].astype(bool)]=[.05,.90,.05]
        candidate=softmax(a['logits_full_512']/profile['temperature'],axis=1)
        reports[cohort]={'archived_runtime':metrics(original,a['labels'],cal['hallucination_threshold']),
                        'archived_candidate':metrics(candidate,a['labels'],profile['hallucination_threshold']),
                        'truncated_original':int((a['length_lexical_256']>256).sum()),
                        'truncated_candidate':int((a['length_full_512']>512).sum())}
    distributions={};keys={};groups={};pooled=defaultdict(set)
    for split in ['train','dev','test']:
        p=ROOT/f'halluciguard_detector/data/processed/{split}.jsonl';inputs.append(p)
        rows=lines(p);pairs=[tuple(' '.join(re.findall(r'\w+',r[k].casefold())) for k in ['claim','evidence']) for r in rows]
        keys[split]=set(pairs);groups[split]={r.get('source_id') for r in rows}
        for pair,r in zip(pairs,rows):pooled[pair].add(r.get('label_id',r.get('label')))
        distributions[split]={'n':len(rows),'labels':dict(Counter(str(r.get('label_id',r.get('label'))) for r in rows)),
                               'repeated_normalized_occurrences':len(pairs)-len(keys[split]),'groups':len(groups[split])}
    overlap={a+'_'+b:{'normalized_pairs':len(keys[a]&keys[b]),'source_groups':len(groups[a]&groups[b])} for a,b in [('train','dev'),('train','test'),('dev','test')]}
    save(RUN/'detector/metrics.json',reports)
    save(RUN/'detector/data_audit.json',{'partitions':distributions,'overlap':overlap,'conflicting_normalized_pairs':sum(len(v)>1 for v in pooled.values()),
        'limitation':'Archive logits lack row IDs. Do not join their predictions to current rows by position.'})
    save(RUN/'detector/configuration.json',provenance(__file__,inputs,kind='reaggregation_of_saved_predictions',api_calls=0,model_hash=sha(checkpoint/'model.safetensors')))
    print(reports['historical_logits']);print(distributions,overlap)

if __name__=='__main__':main()
