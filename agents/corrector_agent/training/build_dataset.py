import os
import json
import random

DATASETS_DIR = r"C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets"
OUTPUT_DIR = "training/data"

def process_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records = []
    
    files = ["halluciguard_dataset.jsonl", "halluciguard_ragtruth.jsonl", "halluciguard_truthfulqa.jsonl"]
    
    for filename in files:
        filepath = os.path.join(DATASETS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filepath}, does not exist.")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    all_records.append(record)
                except Exception as e:
                    print(f"Error parsing line in {filename}: {e}")

    # Deduplicate based on query + original response
    unique_records = {}
    for r in all_records:
        query_val = r.get("query", "")
        orig_val = r.get("original_response", "")
        
        # If both are somehow missing, ensure we don't collapse them all into a single empty key
        key = str(query_val) + str(orig_val)
        if not key:
            key = str(id(r))
            
        if key not in unique_records:
            unique_records[key] = r

    records = list(unique_records.values())
    random.shuffle(records)

    # In a real build_dataset script, we'd invoke PromptBuilder and label generation here
    # Since we can't fully execute Planner without structured JudgePayload natively here,
    # we simulate the transformation for the sake of the script structure.
    
    formatted_dataset = []
    for r in records:
        # Pseudo-formatting to match the expected format for Qwen
        query = r.get("query", "")
        orig_resp = r.get("original_response", "")
        evidence = r.get("verified_evidence", [])
        
        # We craft a naive prompt & target for training since we are synthesizing.
        # This mirrors the logic in the target Kotlin client.
        prompt = f"=== USER QUERY ===\n{query}\n\n=== ORIGINAL LLM RESPONSE ===\n{orig_resp}\n\n=== VERIFIED CLAIMS (MUST PRESERVE EXACTLY) ===\n(None)\n\n=== CLAIMS REQUIRING REWRITING / CORRECTION ===\n1. [ID: c1] \"{orig_resp}\"\n   Issue: {r.get('judge_reason', 'Hallucinated')}\n"
        if evidence:
            prompt += "   Verified Supporting Evidence:\n"
            for ev in evidence:
                prompt += f"     - [Source: {ev.get('source', 'Src')}] \"{ev.get('text', '')}\"\n"
        
        prompt += "\n=== OUTPUT MANDATE ===\nReturn ONLY the final, complete corrected response text. Do not include markdown code fence wrappers, preambles, or conversational meta-commentary."
        
        if evidence:
            target = evidence[0].get("text", "")
        else:
            target = "Current evidence is insufficient to support this claim."
            
        formatted_dataset.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]
        })
        
    num_records = len(formatted_dataset)
    train_split = int(num_records * 0.8)
    val_split = int(num_records * 0.9)
    
    train_data = formatted_dataset[:train_split]
    val_data = formatted_dataset[train_split:val_split]
    test_data = formatted_dataset[val_split:]
    
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_data, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))
    
    print(f"Processed {num_records} records. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    process_datasets()
