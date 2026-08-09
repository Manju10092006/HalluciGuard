import os
import json
import urllib.request
import urllib.error
import time

def call_gemini_batch(batch_records, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    
    system_instruction = "You are an auditor. For each record, check if the Answer is a semantically correct and plausible answer to the Query, based strictly on the Evidence. Flag cases where the Answer is a related term but technically incorrect (e.g., asked for genus but answered with family, asked for city but answered with state, asked for actor but answered with character). Return a JSON list of objects containing only the 'index' of flagged records and a short 'reason'."
    
    prompt = "Here is the batch of records to audit:\n"
    for r in batch_records:
        prompt += f"\n--- Record {r['index']} ---\nQuery: {r['query']}\nEvidence: {r['evidence']}\nAnswer: {r['answer']}\n"
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                candidates = res_data.get('candidates', [])
                if candidates:
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if parts:
                        text = parts[0].get('text', '').strip()
                        return json.loads(text)
            return []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
            else:
                print(f"HTTPError: {e.code} - {e.read().decode('utf-8')}")
                return []
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return []
    return []

def main():
    api_key = os.environ.get("API_KEY", "YOUR_API_KEY")
    
    files = ['halluciguard_dataset.jsonl', 'halluciguard_ragtruth.jsonl', 'halluciguard_truthfulqa.jsonl']
    dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
    
    records_to_audit = []
    
    idx_counter = 0
    file_mapping = {}
    
    for fname in files:
        path = os.path.join(dir_path, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                record = json.loads(line)
                target = record.get('target', {})
                inp = record.get('input', {})
                verdict = inp.get('judge', {}).get('overall_verdict', '').lower()
                
                has_replace = False
                if isinstance(target, dict) and 'actions' in target:
                    for action in target['actions']:
                        if isinstance(action, dict) and action.get('type') == 'replace':
                            has_replace = True
                
                if verdict == 'supported' and not has_replace:
                    ev_texts = []
                    claims = inp.get('judge', {}).get('claims', [])
                    for c in claims:
                        for ev in c.get('evidence', []):
                            if ev.get('text'):
                                ev_texts.append(ev.get('text'))
                    
                    ev_text = " ".join(ev_texts)
                    if not ev_text: continue
                    
                    records_to_audit.append({
                        'index': idx_counter,
                        'file': fname,
                        'line': line_idx,
                        'query': inp.get('user_prompt', ''),
                        'evidence': ev_text,
                        'answer': inp.get('llm_response', '')
                    })
                    file_mapping[idx_counter] = {'file': fname, 'line': line_idx}
                    idx_counter += 1

    print(f"Total supported records to audit: {len(records_to_audit)}")
    
    batch_size = 20
    flagged = []
    
    for i in range(0, len(records_to_audit), batch_size):
        batch = records_to_audit[i:i+batch_size]
        print(f"Auditing batch {i//batch_size + 1} ({len(batch)} records)...")
        res = call_gemini_batch(batch, api_key)
        
        for item in res:
            idx = item.get('index')
            if idx is not None and idx in file_mapping:
                flagged.append({
                    'index': idx,
                    'file': file_mapping[idx]['file'],
                    'line': file_mapping[idx]['line'],
                    'reason': item.get('reason', '')
                })
        
        time.sleep(13.0)

    print(f"\nAudit complete. Found {len(flagged)} flagged records.")
    
    os.makedirs('training/data', exist_ok=True)
    with open('training/data/flagged_supported.json', 'w', encoding='utf-8') as f:
        json.dump(flagged, f, indent=2)
    print("Saved flagged records to training/data/flagged_supported.json")

if __name__ == "__main__":
    main()
