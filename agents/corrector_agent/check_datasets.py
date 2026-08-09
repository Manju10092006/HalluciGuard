import json
import os
import glob

DATASETS_DIR = r"C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets"

print(f"Files in {DATASETS_DIR}:")
for filepath in glob.glob(os.path.join(DATASETS_DIR, '*.jsonl')):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): count += 1
    print(f"{os.path.basename(filepath)}: {count} records")

    # Let's peek at the first record's keys
    with open(filepath, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
        print(f"  Keys: {list(first.keys())}")
