"""Pair independent completed annotations. Blank fields never count as agreement."""
import argparse,csv,json
from sklearn.metrics import cohen_kappa_score

def compare(a,b,field='support_label'):
    left={r['item_id']:r for r in a};right={r['item_id']:r for r in b}
    if len(left)!=len(a) or len(right)!=len(b):raise ValueError('Duplicate item IDs')
    pairs=[(k,left[k][field],right[k][field]) for k in sorted(left.keys()&right.keys()) if left[k][field].strip() and right[k][field].strip()]
    if not pairs:return {'completed_pairs':0,'agreement':None,'cohen_kappa':None,'status':'pending'}
    x=[a for _,a,_ in pairs];y=[b for _,_,b in pairs]
    kappa=float(cohen_kappa_score(x,y)) if len(set(x+y))>1 else None
    return {'completed_pairs':len(pairs),'agreement':sum(a==b for _,a,b in pairs)/len(pairs),
            'cohen_kappa':kappa,'disagreements':[k for k,a,b in pairs if a!=b],
            'unpaired_ids':sorted(left.keys()^right.keys()),'status':'requires_adjudication'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('annotator_a');ap.add_argument('annotator_b');ap.add_argument('--field',default='support_label');args=ap.parse_args()
    with open(args.annotator_a,encoding='utf8',newline='') as f:a=list(csv.DictReader(f))
    with open(args.annotator_b,encoding='utf8',newline='') as f:b=list(csv.DictReader(f))
    print(json.dumps(compare(a,b,args.field),indent=2,allow_nan=False))
