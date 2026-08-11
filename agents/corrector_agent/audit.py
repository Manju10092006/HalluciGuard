import json
import os

DATASETS_DIR = r"C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets"
files = ["halluciguard_dataset.jsonl", "halluciguard_ragtruth.jsonl", "halluciguard_truthfulqa.jsonl"]

total = 0
disclaimers = 0
evidence_present_disclaimer = 0

for filename in files:
    filepath = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                total += 1
                evidence = record.get("verified_evidence", [])
                
                # Check what the target would be
                # In build_dataset.py:
                # if evidence: target = evidence[0].text
                # else: target = "Current evidence is insufficient to support this claim."
                if evidence:
                    pass
                else:
                    disclaimers += 1
                    
            except Exception as e:
                pass

print(f"Total records: {total}")
print(f"Disclaimers (no evidence): {disclaimers}")
