import os
import json
import random

DIR_PATH = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
RAW_FILES = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']
FLAGGED_SUPPORTED_FILE = 'training/data/flagged_supported.json'
STAGE1_FILE = 'training/data/shape_b_llm_synthesized_stage1.jsonl'
STAGE2_FILE = 'training/data/shape_b_llm_synthesized_stage2.jsonl'
OUTPUT_DIR = 'training/data'

def build_prompt(record):
    inp = record.get('input', {})
    query = inp.get('user_prompt', '')
    orig = inp.get('llm_response', '')
    judge = inp.get('judge', {})
    claims = judge.get('claims', [])
    
    prompt = f'=== USER QUERY ===\n{query}\n\n=== ORIGINAL LLM RESPONSE ===\n{orig}\n\n'
    
    verified_claims = [c for c in claims if c.get('status', '').lower() == 'supported']
    unsupported_claims = [c for c in claims if c.get('status', '').lower() in ['unsupported', 'hallucinated']]
    
    prompt += '=== VERIFIED CLAIMS (MUST PRESERVE EXACTLY) ===\n'
    if verified_claims:
        for i, c in enumerate(verified_claims, 1):
            prompt += f'{i}. "{c.get("claim", "")}"\n'
            for ev in c.get('evidence', []):
                prompt += f'   - [Source: {ev.get("source", "Src")}] "{ev.get("text", "")}"\n'
    else:
        prompt += '(None)\n'
        
    prompt += '\n=== CLAIMS REQUIRING REWRITING / CORRECTION ===\n'
    if unsupported_claims:
        for i, c in enumerate(unsupported_claims, 1):
            prompt += f'{i}. "{c.get("claim", "")}"\n   Issue: {c.get("reason", "Hallucinated")}\n'
            for ev in c.get('evidence', []):
                prompt += f'   Verified Supporting Evidence:\n     - [Source: {ev.get("source", "Src")}] "{ev.get("text", "")}"\n'
    else:
        prompt += '(None)\n'
        
    prompt += '\n=== OUTPUT MANDATE ===\nReturn ONLY the final, complete corrected response text. Do not include markdown code fence wrappers, preambles, or conversational meta-commentary.'
    return prompt

def get_flagged_supported_indices():
    indices = set()
    if not os.path.exists(FLAGGED_SUPPORTED_FILE): return indices
    with open(FLAGGED_SUPPORTED_FILE, 'r', encoding='utf-8') as f:
        flagged = json.load(f)
    for item in flagged:
        fname = item['file']
        line_idx = item['line'] - 1
        indices.add((fname, line_idx))
    return indices

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

def main():
    flagged_supported = get_flagged_supported_indices()
    
    # Pre-calculate prompts for flagged_supported so we can drop them from Shape B files if needed
    flagged_supported_prompts = set()
    for fname, idx in flagged_supported:
        path = os.path.join(DIR_PATH, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                record = json.loads(lines[idx])
                flagged_supported_prompts.add(build_prompt(record))
                
    final_records = []
    
    # 1. Load Shape A records
    shape_a_count = 0
    dropped_corrupted = 0
    dropped_semantically_wrong_a = 0
    
    for fname in RAW_FILES:
        path = os.path.join(DIR_PATH, fname)
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip(): continue
                record = json.loads(line)
                target = record.get('target', {})
                
                # Check for semantically-wrong
                if (fname, idx) in flagged_supported:
                    if isinstance(target, dict) and 'actions' in target and 'analysis' in target:
                        dropped_semantically_wrong_a += 1
                    continue
                    
                # Skip malformed (5)
                if not isinstance(target, dict): continue
                
                # Check if it's Shape A
                if 'actions' in target and 'analysis' in target:
                    # Check for corrupted
                    if is_corrupted_shape_a(record):
                        dropped_corrupted += 1
                        continue
                        
                    shape_a_count += 1
                    prompt = build_prompt(record)
                    orig = record['input'].get('llm_response', '')
                    corrected = orig
                    for repl in target.get('actions', {}).get('replace', []):
                        old_txt = repl.get('old', '')
                        new_txt = repl.get('new', '')
                        if old_txt and old_txt in corrected:
                            corrected = corrected.replace(old_txt, new_txt)
                            
                    msg = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": corrected}
                        ]
                    }
                    final_records.append(msg)
                        
    print(f"Dropped {dropped_corrupted} corrupted Shape A records.")
    print(f"Dropped {dropped_semantically_wrong_a} semantically-wrong Shape A records.")
    print(f"Processed {shape_a_count} valid Shape A records.")
    
    # 2. Load Shape B synthesized records
    shape_b_count = 0
    dropped_semantically_wrong_b = 0
    seen_shape_b_prompts = set()
    for sfile in [STAGE1_FILE, STAGE2_FILE]:
        if not os.path.exists(sfile): continue
        with open(sfile, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                prompt = record['messages'][0]['content']
                if prompt in flagged_supported_prompts:
                    dropped_semantically_wrong_b += 1
                    continue
                    
                if prompt not in seen_shape_b_prompts:
                    seen_shape_b_prompts.add(prompt)
                    final_records.append(record)
                    shape_b_count += 1
                    
    print(f"Dropped {dropped_semantically_wrong_b} semantically-wrong Shape B records.")
    print(f"Processed {shape_b_count} valid Shape B records from synthesized files.")
    print(f"Total dataset size: {len(final_records)}")
    
    # 3. Shuffle and split 90/10
    random.seed(42)
    random.shuffle(final_records)
    
    split_idx = int(len(final_records) * 0.9)
    train_data = final_records[:split_idx]
    val_data = final_records[split_idx:]
    
    with open(os.path.join(OUTPUT_DIR, 'train.jsonl'), 'w', encoding='utf-8') as f:
        for r in train_data:
            f.write(json.dumps(r) + '\n')
            
    with open(os.path.join(OUTPUT_DIR, 'val.jsonl'), 'w', encoding='utf-8') as f:
        for r in val_data:
            f.write(json.dumps(r) + '\n')
            
    print(f"Saved {len(train_data)} to train.jsonl and {len(val_data)} to val.jsonl")

if __name__ == '__main__':
    main()
