import os
import json
import urllib.request
import urllib.error
import argparse
import sys
import time

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

def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    
    system_instruction = (
        "Using ONLY the facts stated in this evidence text, write a concise, natural corrected answer to the query. "
        "Do not add any information not present in the evidence. "
        "If the evidence does not provide enough information to fully resolve a comparison or claim, say so explicitly rather than asserting a conclusion the evidence doesn't fully support."
    )
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1
        }
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        candidates = res_data.get('candidates', [])
        if candidates:
            parts = candidates[0].get('content', {}).get('parts', [])
            if parts:
                return parts[0].get('text', '').strip()
    return ""

def call_openai(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    
    system_instruction = (
        "Using ONLY the facts stated in this evidence text, write a concise, natural corrected answer to the query. "
        "Do not add any information not present in the evidence. "
        "If the evidence does not provide enough information to fully resolve a comparison or claim, say so explicitly rather than asserting a conclusion the evidence doesn't fully support."
    )
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                 headers={
                                     'Content-Type': 'application/json',
                                     'Authorization': f'Bearer {api_key}'
                                 })
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        choices = res_data.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '').strip()
    return ""

def synthesize_with_retry(synthesize_fn, prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return synthesize_fn(prompt)
        except urllib.error.HTTPError as e:
            err_msg = f"HTTPError: {e.code} - {e.read().decode('utf-8')}"
        except Exception as e:
            err_msg = f"Error: {str(e)}"
            
        if attempt < max_retries:
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s...
        else:
            raise Exception(err_msg)

def log_failure(failures_file, index, record, reason):
    with open(failures_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            "index": index,
            "reason": reason,
            "user_prompt": record.get("input", {}).get("user_prompt", "")
        }) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Synthesize Shape B records using an LLM API.")
    parser.add_argument("--start", type=int, default=0, help="Starting index to process")
    parser.add_argument("--limit", type=int, default=177, help="Number of records to process")
    parser.add_argument("--out", type=str, default="training/data/shape_b_llm_synthesized_run.jsonl", help="Output JSONL path")
    parser.add_argument("--failures", type=str, default="training/data/stage1_failures.jsonl", help="Output for failed records")
    parser.add_argument("--retry-failures", action="store_true", help="Retry failed records from failures file")
    parser.add_argument("--max-calls", type=int, default=18, help="Maximum number of API calls to make in one run")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        print("Detected GEMINI_API_KEY. Using Gemini API.")
        synthesize_fn = lambda p: call_gemini(p, gemini_key)
    elif openai_key:
        print("Detected OPENAI_API_KEY. Using OpenAI API.")
        synthesize_fn = lambda p: call_openai(p, openai_key)
    else:
        print("Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY found in environment.")
        print("Please set one of these variables before running this script.")
        sys.exit(1)

    dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
    fname = 'halluciguard_dataset.jsonl'
    path = os.path.join(dir_path, fname)
    
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        sys.exit(1)

    shape_b_records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            target = record.get('target', {})
            if not isinstance(target, dict): continue
            if 'actions' in target and 'analysis' in target: continue # skip Shape A
            shape_b_records.append(record)

    already_processed_indices = set()
    if os.path.exists(args.out):
        prompt_to_idx = {build_prompt(r): idx for idx, r in enumerate(shape_b_records)}
        with open(args.out, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    out_rec = json.loads(line)
                    msgs = out_rec.get('messages', [])
                    if msgs and len(msgs) > 0:
                        user_msg = msgs[0].get('content', '')
                        if user_msg in prompt_to_idx:
                            already_processed_indices.add(prompt_to_idx[user_msg])
                except Exception:
                    pass
        print(f"Found {len(already_processed_indices)} already processed records in {args.out}.")
        print(f"Already processed indices: {sorted(list(already_processed_indices))}")

    failed_indices_map = {}
    if args.retry_failures:
        if os.path.exists(args.failures):
            with open(args.failures, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    failed_indices_map[rec['index']] = rec
            records_to_process = [(idx, shape_b_records[idx]) for idx in failed_indices_map.keys() if idx < len(shape_b_records) and idx not in already_processed_indices]
        else:
            print("No failures file found to retry.")
            sys.exit(0)
    else:
        records_to_process = [(args.start + i, shape_b_records[args.start + i]) for i in range(args.limit) if args.start + i < len(shape_b_records) and (args.start + i) not in already_processed_indices]
    
    print(f"Total Shape B records in source: {len(shape_b_records)}")
    print(f"Processing {len(records_to_process)} records...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    api_calls_made = 0
    successful_indices = set()
    new_failed_indices = set()
    
    file_mode = 'a'
    
    with open(args.out, file_mode, encoding='utf-8') as out:
        for actual_index, r in records_to_process:
            verdict = r.get('input',{}).get('judge',{}).get('overall_verdict','').lower()
            prompt = build_prompt(r)
            
            if verdict == 'hallucinated':
                if api_calls_made >= args.max_calls:
                    print(f"Reached max-calls limit of {args.max_calls}. Stopping cleanly.")
                    break
                
                print(f"Synthesizing record {actual_index} (Hallucinated)...")
                api_calls_made += 1
                try:
                    target_resp = synthesize_with_retry(synthesize_fn, prompt)
                    if not target_resp:
                        print(f"Empty response for record {actual_index}. Skipping and logging.")
                        if not args.retry_failures:
                            log_failure(args.failures, actual_index, r, "empty_response")
                        new_failed_indices.add(actual_index)
                        continue
                except Exception as e:
                    print(f"Failed to synthesize record {actual_index}: {e}. Skipping and logging.")
                    if not args.retry_failures:
                        log_failure(args.failures, actual_index, r, str(e))
                    new_failed_indices.add(actual_index)
                    continue
                time.sleep(13.0) # rate limit buffer
            else:
                target_resp = r.get('input',{}).get('llm_response', '')
                
            out.write(json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_resp}
                ]
            }) + '\n')
            successful_indices.add(actual_index)

    print(f"\nDone! Made {api_calls_made} API calls. Processed and wrote {len(successful_indices)} records.")
    
    if args.retry_failures:
        remaining_failures = []
        for idx, rec in failed_indices_map.items():
            if idx not in successful_indices:
                remaining_failures.append(rec)
        
        with open(args.failures, 'w', encoding='utf-8') as f:
            for rec in remaining_failures:
                f.write(json.dumps(rec) + '\n')
        
        print(f"Failures updated: {len(successful_indices)} successes removed, {len(remaining_failures)} failures remaining for future retries.")

if __name__ == "__main__":
    main()
