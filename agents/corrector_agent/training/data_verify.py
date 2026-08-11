import json
import os

DIR_PATH = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
RAW_FILES = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']
FLAGGED_SUPPORTED_FILE = 'training/data/flagged_supported.json'
STAGE1_FILE = 'training/data/shape_b_llm_synthesized_stage1.jsonl'
STAGE2_FILE = 'training/data/shape_b_llm_synthesized_stage2.jsonl'
TRAIN_FILE = 'training/data/train.jsonl'

def get_flagged_supported():
    with open(FLAGGED_SUPPORTED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_corrupted_shape_a(record):
    target = record.get('target', {})
    inp = record.get('input', {})
    verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
    
    if not isinstance(target, dict): return False
    if 'actions' in target and 'analysis' in target:
        replaces = target.get('actions', {}).get('replace', [])
        if replaces:
            if verdict == 'supported':
                return True
            else:
                supported_claims = [c.get('claim', '') for c in inp.get('judge', {}).get('claims', []) if c.get('status') == 'supported']
                for repl in replaces:
                    old_txt = repl.get('old', '')
                    if any(old_txt in c for c in supported_claims if old_txt):
                        return True
    return False

flagged_supported_items = get_flagged_supported()
flagged_supported_set = {(item['file'], item['line']-1): item['reason'] for item in flagged_supported_items}

corrupted_list = []
sem_wrong_a = []
sem_wrong_b = []
overlap_list = []

for fname in RAW_FILES:
    path = os.path.join(DIR_PATH, fname)
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            record = json.loads(line)
            query = record['input']['user_prompt']
            
            is_sem_wrong = (fname, idx) in flagged_supported_set
            is_corrupted = False
            
            target = record.get('target', {})
            if isinstance(target, dict) and 'actions' in target and 'analysis' in target:
                is_shape_a = True
                is_corrupted = is_corrupted_shape_a(record)
            else:
                is_shape_a = False
                
            if is_corrupted and is_sem_wrong:
                overlap_list.append({'query': query, 'file': fname, 'line': idx+1, 'reason': flagged_supported_set[(fname, idx)]})
                
            if is_corrupted:
                corrupted_list.append({'query': query, 'file': fname, 'line': idx+1})
                
            if is_sem_wrong:
                if is_shape_a:
                    sem_wrong_a.append({'query': query, 'file': fname, 'line': idx+1, 'reason': flagged_supported_set[(fname, idx)]})
                else:
                    sem_wrong_b.append({'query': query, 'file': fname, 'line': idx+1, 'reason': flagged_supported_set[(fname, idx)]})

print("=== OVERLAP (CORRUPTED + SEMANTICALLY WRONG) ===")
for r in overlap_list:
    print(f"File: {r['file']}, Line: {r['line']}\nQuery: {r['query']}\nReason: {r['reason']}\n")

print("\n=== 14 CORRUPTED SHAPE A RECORDS ===")
for r in corrupted_list:
    print(f"File: {r['file']}, Line: {r['line']}\nQuery: {r['query']}")

print("\n=== 9 SEMANTICALLY WRONG SHAPE A RECORDS ===")
for r in sem_wrong_a:
    print(f"File: {r['file']}, Line: {r['line']}\nQuery: {r['query']}\nReason: {r['reason']}\n")

print("\n=== 3 SEMANTICALLY WRONG SHAPE B RECORDS ===")
for r in sem_wrong_b:
    print(f"File: {r['file']}, Line: {r['line']}\nQuery: {r['query']}\nReason: {r['reason']}\n")

print("\n=== VERIFYING DEDUPLICATION OF STAGE 1 ===")
stage1_prompts = []
if os.path.exists(STAGE1_FILE):
    with open(STAGE1_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            stage1_prompts.append(record['messages'][0]['content'])

duplicates = []
seen = set()
for p in stage1_prompts:
    if p in seen:
        duplicates.append(p)
    seen.add(p)
print(f"Total Stage 1 lines: {len(stage1_prompts)}")
print(f"Distinct Stage 1 prompts: {len(seen)}")
print(f"Number of exact duplicate prompts in Stage 1: {len(duplicates)}")
if duplicates:
    # Print the first few characters of a duplicate prompt to show they are identical
    print(f"Sample duplicate prompt starts with:\n{duplicates[0][:100]}...\n")

print("\n=== 5 RANDOM SAMPLES FROM TRAIN.JSONL ===")
import random
random.seed(123)
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_lines = f.readlines()
samples = random.sample(train_lines, 5)
for i, s in enumerate(samples, 1):
    record = json.loads(s)
    print(f"--- Sample {i} ---")
    for msg in record['messages']:
        print(f"Role: {msg['role']}\n{msg['content']}\n")
