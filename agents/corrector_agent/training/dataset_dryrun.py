import os
import json

dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
files = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']

shape_a_examples = []
shape_b_examples = []
malformed = []

total_usable = 0
hallucinated_cnt = 0
supported_cnt = 0

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

for fname in files:
    path = os.path.join(dir_path, fname)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            target = record.get('target', {})
            inp = record.get('input', {})
            verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
            
            # Identify shape
            if not isinstance(target, dict):
                malformed.append((inp.get('user_prompt', ''), target))
                continue
                
            if 'actions' in target and 'analysis' in target:
                # Shape A
                total_usable += 1
                if verdict == 'hallucinated': hallucinated_cnt += 1
                elif verdict == 'supported': supported_cnt += 1
                
                if len(shape_a_examples) < 3:
                    prompt = build_prompt(record)
                    orig = inp.get('llm_response', '')
                    corrected = orig
                    for repl in target.get('actions', {}).get('replace', []):
                        old_txt = repl.get('old', '')
                        new_txt = repl.get('new', '')
                        if old_txt and old_txt in corrected:
                            corrected = corrected.replace(old_txt, new_txt)
                    shape_a_examples.append({'prompt': prompt, 'target': corrected})
            else:
                # Shape B (mirrors input)
                total_usable += 1
                if verdict == 'hallucinated': hallucinated_cnt += 1
                elif verdict == 'supported': supported_cnt += 1
                
                if len(shape_b_examples) < 3:
                    prompt = build_prompt(record)
                    orig = inp.get('llm_response', '')
                    corrected = orig
                    if verdict == 'hallucinated':
                        claims = inp.get('judge', {}).get('claims', [])
                        for c in claims:
                            if c.get('status') == 'unsupported':
                                c_text = c.get('claim', '')
                                ev = c.get('evidence', [])
                                if ev and c_text and c_text in corrected:
                                    corrected = corrected.replace(c_text, ev[0].get('text', ''))
                    shape_b_examples.append({'prompt': prompt, 'target': corrected})

print('--- SHAPE A ---')
for ex in shape_a_examples:
    print('USER_MESSAGE:\n' + ex['prompt'] + '\nTARGET:\n' + ex['target'] + '\n' + '='*40)
print('--- SHAPE B ---')
for ex in shape_b_examples:
    print('USER_MESSAGE:\n' + ex['prompt'] + '\nTARGET:\n' + ex['target'] + '\n' + '='*40)
print('--- MALFORMED ---')
for m in malformed:
    print(m)
print(f'TOTAL: {total_usable}, Hallucinated: {hallucinated_cnt}, Supported: {supported_cnt}')
