import json
import os
import glob

DATASETS_DIR = r"C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets"

all_records = []
for filepath in glob.glob(os.path.join(DATASETS_DIR, '*.jsonl')):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

unique_records = {}
for r in all_records:
    query_val = r.get("input", {}).get("user_prompt", "")
    orig_val = r.get("input", {}).get("llm_response", "")
    
    key = str(query_val) + str(orig_val)
    if not key:
        key = str(id(r))
        
    if key not in unique_records:
        unique_records[key] = r

print(f"Total records before deduplication: {len(all_records)}")
print(f"Total records after correct deduplication: {len(unique_records)}")
