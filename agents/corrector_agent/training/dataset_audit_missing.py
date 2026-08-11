import os
import json

dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
files = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']

flagged_15 = []
malformed = []

# First pass: find the 5 malformed and 15 flagged records so we can exclude them
for fname in files:
    path = os.path.join(dir_path, fname)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            target = record.get('target', {})
            inp = record.get('input', {})
            verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
            query = inp.get('user_prompt', '')
            
            if not isinstance(target, dict):
                malformed.append(query)
                continue
                
            if 'actions' in target and 'analysis' in target:
                replaces = target.get('actions', {}).get('replace', [])
                if replaces:
                    flagged = False
                    if verdict == 'supported':
                        flagged = True
                    else:
                        supported_claims = [c.get('claim', '') for c in inp.get('judge', {}).get('claims', []) if c.get('status') == 'supported']
                        for repl in replaces:
                            old_txt = repl.get('old', '')
                            if any(old_txt in c for c in supported_claims if old_txt):
                                flagged = True
                                break
                    if flagged:
                        flagged_15.append(query)

missing_old_text = []
hallucinated_fail_cnt = 0
supported_fail_cnt = 0

for fname in files:
    path = os.path.join(dir_path, fname)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            target = record.get('target', {})
            inp = record.get('input', {})
            query = inp.get('user_prompt', '')
            
            if query in malformed or query in flagged_15:
                continue
                
            orig = inp.get('llm_response', '')
            verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
            
            failed_replacements = []
            
            if 'actions' in target and 'analysis' in target:
                # Shape A
                replaces = target.get('actions', {}).get('replace', [])
                for repl in replaces:
                    old_txt = repl.get('old', '')
                    if old_txt and old_txt not in orig:
                        failed_replacements.append(old_txt)
            else:
                # Shape B
                if verdict == 'hallucinated':
                    claims = inp.get('judge', {}).get('claims', [])
                    for c in claims:
                        if c.get('status') == 'unsupported':
                            c_text = c.get('claim', '')
                            if c_text and c_text not in orig:
                                failed_replacements.append(c_text)
                                
            if failed_replacements:
                if verdict == 'hallucinated': hallucinated_fail_cnt += 1
                elif verdict == 'supported': supported_fail_cnt += 1
                
                missing_old_text.append({
                    'query': query,
                    'original': orig,
                    'verdict': verdict,
                    'failed_old_texts': failed_replacements
                })

print(f"Total failing records: {len(missing_old_text)}")
print(f"Hallucinated fails: {hallucinated_fail_cnt}")
print(f"Supported fails: {supported_fail_cnt}")
print("--- 5 EXAMPLES ---")
for r in missing_old_text[:5]:
    print(f"QUERY: {r['query']}")
    print(f"ORIGINAL: {r['original']}")
    print(f"VERDICT: {r['verdict']}")
    print(f"FAILED OLD TEXTS: {r['failed_old_texts']}")
    print("-" * 40)
