import json

FILES = ['training/data/train.jsonl', 'training/data/val.jsonl']
UNINFORMATIVE_PHRASES = [
    "i have no comment",
    "no comment",
]

def is_uninformative(text):
    t = text.lower().strip()
    if t == "": return True
    if any(p in t for p in UNINFORMATIVE_PHRASES): return True
    return False

ungrounded_records = []

for fpath in FILES:
    with open(fpath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            record = json.loads(line)
            prompt = record['messages'][0]['content']
            target = record['messages'][1]['content']
            
            # The prompt contains lines with Verified Supporting Evidence:
            # Let's extract them
            lines = prompt.split('\n')
            evidence_lines = []
            in_evidence = False
            
            # Note: A prompt can have multiple "Verified Supporting Evidence:" blocks
            # if there are multiple claims.
            # But the user asked to check if the evidence text is a non-answer.
            # Let's extract all sources across the whole prompt.
            for p_line in lines:
                if '[Source:' in p_line:
                    parts = p_line.split(']', 1)
                    if len(parts) > 1:
                        evidence_text = parts[1].strip()
                        if evidence_text.startswith('"') and evidence_text.endswith('"'):
                            evidence_text = evidence_text[1:-1]
                        evidence_lines.append(evidence_text)

            # Check if ANY of the evidence lines are uninformative
            # We want records where ALL provided evidence is uninformative, OR
            # AT LEAST ONE claim has uninformative evidence but the target still asserts a fact?
            # The user asked: "count how many records across the full 738-record dataset have evidence text that is a non-answer ... while the assistant target still asserts a specific concrete fact."
            # So if ANY evidence line is uninformative AND the target asserts a fact.
            # Wait, if there are multiple evidence lines and one of them is "I have no comment", it might be ungrounded.
            if len(evidence_lines) > 0 and any(is_uninformative(e) for e in evidence_lines):
                # The target shouldn't be just "I have no comment"
                if not is_uninformative(target):
                    ungrounded_records.append({
                        'query': prompt.split('=== USER QUERY ===')[1].split('===')[0].strip(),
                        'evidence': evidence_lines,
                        'target': target
                    })

print(f"Total ungrounded records found: {len(ungrounded_records)}")
print(f"Percentage: {len(ungrounded_records)/738*100:.2f}%")
if len(ungrounded_records) > 0:
    print("Examples:")
    for i, r in enumerate(ungrounded_records[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {r['query']}")
        print(f"Evidence: {r['evidence']}")
        print(f"Assistant Target: {r['target']}")
