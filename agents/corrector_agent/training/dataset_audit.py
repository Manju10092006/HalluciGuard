import os
import json

dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
files = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']

flagged_records = []

for fname in files:
    path = os.path.join(dir_path, fname)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            target = record.get('target', {})
            inp = record.get('input', {})
            verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
            
            if not isinstance(target, dict):
                continue
                
            if 'actions' in target and 'analysis' in target:
                # Shape A
                replaces = target.get('actions', {}).get('replace', [])
                if not replaces: continue
                
                flagged = False
                reason = ""
                
                if verdict == 'supported':
                    flagged = True
                    reason = "Verdict is supported but target has replacement actions"
                else:
                    # check if it replaces a supported claim
                    supported_claims = [c.get('claim', '') for c in inp.get('judge', {}).get('claims', []) if c.get('status') == 'supported']
                    for repl in replaces:
                        old_txt = repl.get('old', '')
                        if any(old_txt in c for c in supported_claims if old_txt):
                            flagged = True
                            reason = f"Replaces text belonging to a supported claim: '{old_txt}'"
                            break
                            
                if flagged:
                    flagged_records.append({
                        'query': inp.get('user_prompt', ''),
                        'original': inp.get('llm_response', ''),
                        'verdict': verdict,
                        'reason': reason,
                        'actions': replaces
                    })

print(f"Total flagged records: {len(flagged_records)}")
print("--- 5 EXAMPLES ---")
for r in flagged_records[:5]:
    print(f"QUERY: {r['query']}")
    print(f"ORIGINAL: {r['original']}")
    print(f"VERDICT: {r['verdict']}")
    print(f"FLAG REASON: {r['reason']}")
    print(f"ACTIONS: {r['actions']}")
    print("-" * 40)
