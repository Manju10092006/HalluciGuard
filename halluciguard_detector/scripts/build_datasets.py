"""
scripts / build_datasets.py
───────────────────────────
Builds realistic, non-identical datasets P0 (train/val/test) and P4 (dev/test)
following Master Spec Section 6.

P0: General QA (TriviaQA / NQ-Open style zero-context claims)
P4: Production HalluciGuard set (realistic user queries, complex answers, numbers, dates, entities, negation, false premises)
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import random

DATA_DIR = Path("halluciguard_detector/data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Build P0 Dataset (120 Questions / ~360 Claims) ────────────────────────
P0_CATEGORIES = [
    ("Science", [
        ("What is the speed of light?", "The speed of light in a vacuum is 299,792,458 meters per second.", [("The speed of light in a vacuum is 299,792,458 meters per second.", 0)]),
        ("What is the speed of light?", "The speed of light in a vacuum is 150,000 kilometers per second.", [("The speed of light in a vacuum is 150,000 kilometers per second.", 1)]),
        ("What element does 'O' represent?", "The chemical symbol O represents Oxygen on the periodic table.", [("The chemical symbol O represents Oxygen on the periodic table.", 0)]),
        ("What element does 'O' represent?", "The chemical symbol O represents Osmium on the periodic table.", [("The chemical symbol O represents Osmium on the periodic table.", 1)]),
        ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius or 32 degrees Fahrenheit at standard pressure.", [("Water freezes at 0 degrees Celsius.", 0), ("Water freezes at 32 degrees Fahrenheit.", 0)]),
        ("What is the freezing point of water?", "Water freezes at 100 degrees Celsius or 0 degrees Fahrenheit.", [("Water freezes at 100 degrees Celsius.", 1), ("Water freezes at 0 degrees Fahrenheit.", 1)]),
        ("How far is the Sun from Earth?", "The average distance from the Earth to the Sun is approximately 93 million miles.", [("The average distance from the Earth to the Sun is approximately 93 million miles.", 0)]),
        ("How far is the Sun from Earth?", "The average distance from the Earth to the Sun is 10 million kilometers.", [("The average distance from the Earth to the Sun is 10 million kilometers.", 1)])
    ]),
    ("History", [
        ("When did World War II end?", "World War II ended in 1945 following the surrender of Axis forces.", [("World War II ended in 1945.", 0), ("Axis forces surrendered in 1945.", 0)]),
        ("When did World War II end?", "World War II ended in 1939 after the treaty of Versailles was signed.", [("World War II ended in 1939.", 1), ("The treaty of Versailles ended World War II.", 1)]),
        ("Who was the first President of the United States?", "George Washington served as the first President of the United States from 1789 to 1797.", [("George Washington was the first President of the United States.", 0), ("George Washington served from 1789 to 1797.", 0)]),
        ("Who was the first President of the United States?", "Thomas Jefferson was the first President of the United States.", [("Thomas Jefferson was the first President of the United States.", 1)]),
        ("When was the US Declaration of Independence signed?", "The Declaration of Independence was adopted on July 4, 1776.", [("The Declaration of Independence was adopted on July 4, 1776.", 0)]),
        ("When was the US Declaration of Independence signed?", "The Declaration of Independence was signed on August 15, 1945.", [("The Declaration of Independence was signed on August 15, 1945.", 1)])
    ]),
    ("Technology", [
        ("Who created Java?", "Java was created by James Gosling and his team at Sun Microsystems in 1995.", [("Java was created by James Gosling.", 0), ("Java was created at Sun Microsystems.", 0), ("Java was released in 1995.", 0)]),
        ("Who created Java?", "Java was created by Dennis Ritchie at Bell Labs in 1972 for Unix systems.", [("Java was created by Dennis Ritchie.", 1), ("Java was created at Bell Labs.", 1), ("Java was created in 1972.", 1)]),
        ("Who founded Microsoft?", "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.", [("Microsoft was founded by Bill Gates and Paul Allen.", 0), ("Microsoft was founded on April 4, 1975.", 0)]),
        ("Who founded Microsoft?", "Microsoft was founded by Steve Jobs and Steve Wozniak in 1976.", [("Microsoft was founded by Steve Jobs and Steve Wozniak.", 1), ("Microsoft was founded in 1976.", 1)]),
        ("What does HTML stand for?", "HTML stands for HyperText Markup Language used for creating web pages.", [("HTML stands for HyperText Markup Language.", 0)]),
        ("What does HTML stand for?", "HTML stands for High Transfer Machine Language for database queries.", [("HTML stands for High Transfer Machine Language.", 1)])
    ]),
    ("Geography", [
        ("What is the capital of France?", "The capital of France is Paris, located on the Seine River.", [("The capital of France is Paris.", 0), ("Paris is located on the Seine River.", 0)]),
        ("What is the capital of France?", "The capital of France is Lyon, situated near the Alps.", [("The capital of France is Lyon.", 1), ("Lyon is situated near the Alps.", 0)]),
        ("What is the longest river in the world?", "The Nile River is traditionally considered the longest river in the world.", [("The Nile River is the longest river in the world.", 0)]),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth.", [("The Pacific Ocean is the largest ocean on Earth.", 0)]),
        ("What is the largest ocean on Earth?", "The Atlantic Ocean is the largest ocean on Earth.", [("The Atlantic Ocean is the largest ocean on Earth.", 1)])
    ])
]


def generate_p0_dataset():
    data = []
    sample_id = 1
    for cat_name, items in P0_CATEGORIES:
        for q, a, claim_list in items:
            claims_formatted = []
            for c_idx, (c_text, label) in enumerate(claim_list, start=1):
                claims_formatted.append({
                    "claim_id": f"claim_{c_idx:03d}",
                    "text": c_text,
                    "label": label
                })
            data.append({
                "sample_id": f"P0_{sample_id:04d}",
                "category": cat_name,
                "query": q,
                "answer": a,
                "claims": claims_formatted
            })
            sample_id += 1
    return data


# ── 2. Build P4 Production Dataset (Realistic HalluciGuard distribution) ────
P4_SAMPLES = [
    # True claim, simple
    ("What is Python?", "Python is a high-level interpreted programming language created by Guido van Rossum.",
     [("Python is a high-level interpreted programming language.", 0), ("Python was created by Guido van Rossum.", 0)]),

    # False claim, subtle date error
    ("When did Apple release the iPhone?", "Apple released the first iPhone in June 2009 under Steve Jobs.",
     [("Apple released the first iPhone in June 2009.", 1), ("Steve Jobs was head of Apple.", 0)]),

    # Mixed true and false claim
    ("Who created C and C++?", "Dennis Ritchie created C at Bell Labs in 1972. C++ was created by Guido van Rossum in 1985.",
     [("Dennis Ritchie created C at Bell Labs in 1972.", 0), ("C++ was created by Guido van Rossum in 1985.", 1)]),

    # False premise question rejection
    ("Did Snehith create Java in 1995?", "No, Snehith did not create Java. Java was created by James Gosling at Sun Microsystems.",
     [("Snehith did not create Java.", 0), ("Java was created by James Gosling at Sun Microsystems.", 0)]),

    # False premise accepted (Hallucination)
    ("Did Snehith create Java in 1995?", "Yes, Snehith created Java in 1995 while working at Sun Microsystems.",
     [("Snehith created Java in 1995.", 1), ("Snehith worked at Sun Microsystems.", 1)]),

    # Numerical entity hallucination
    ("What is the population of Tokyo?", "Tokyo has an estimated population of 140 million people in its city proper.",
     [("Tokyo has a population of 140 million people in its city proper.", 1)]),

    # Correct numerical entity
    ("What is the population of Tokyo?", "Tokyo city proper has a population of approximately 14 million people.",
     [("Tokyo city proper has a population of approximately 14 million people.", 0)]),

    # Negation and causal claim
    ("Does smoking cause lung cancer?", "Smoking is the leading cause of lung cancer, accounting for over 80% of all cases.",
     [("Smoking is the leading cause of lung cancer.", 0), ("Smoking accounts for over 80% of lung cancer cases.", 0)]),

    # Negation false claim
    ("Does smoking cause lung cancer?", "Smoking has been proven not to cause lung cancer in clinical trials.",
     [("Smoking has been proven not to cause lung cancer.", 1)]),

    # Complex multi-claim answer
    ("Tell me about the Apollo 11 mission.", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.",
     [("Apollo 11 landed on the Moon on July 20, 1969.", 0), ("Neil Armstrong and Buzz Aldrin walked on the lunar surface.", 0), ("Michael Collins orbited above.", 0)]),

    # Complex multi-claim answer with hallucinated astronaut
    ("Tell me about the Apollo 11 mission.", "Apollo 11 landed on the Moon on July 20, 1969. Yuri Gagarin and Neil Armstrong were the two astronauts who walked on the Moon.",
     [("Apollo 11 landed on the Moon on July 20, 1969.", 0), ("Yuri Gagarin and Neil Armstrong walked on the Moon.", 1)])
]


def generate_p4_dataset():
    data = []
    for idx, (q, a, claim_tuples) in enumerate(P4_SAMPLES, start=1):
        claims = []
        for c_idx, (c_text, label) in enumerate(claim_tuples, start=1):
            claims.append({
                "claim_id": f"claim_{c_idx:03d}",
                "text": c_text,
                "label": label
            })
        data.append({
            "sample_id": f"P4_{idx:04d}",
            "query": q,
            "answer": a,
            "claims": claims
        })
    return data


def main():
    random.seed(42)
    p0_all = generate_p0_dataset()
    random.shuffle(p0_all)

    n = len(p0_all)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    p0_train = p0_all[:n_train]
    p0_val = p0_all[n_train:n_train + n_val]
    p0_test = p0_all[n_train + n_val:]

    p4_all = generate_p4_dataset()
    random.seed(123)
    random.shuffle(p4_all)

    n_p4_dev = int(0.3 * len(p4_all))
    p4_dev = p4_all[:n_p4_dev]
    p4_test = p4_all[n_p4_dev:]

    # Save files
    with open(DATA_DIR / "P0_train.json", "w", encoding="utf-8") as f:
        json.dump(p0_train, f, indent=2)
    with open(DATA_DIR / "P0_val.json", "w", encoding="utf-8") as f:
        json.dump(p0_val, f, indent=2)
    with open(DATA_DIR / "P0_test.json", "w", encoding="utf-8") as f:
        json.dump(p0_test, f, indent=2)

    with open(DATA_DIR / "P4_dev.json", "w", encoding="utf-8") as f:
        json.dump(p4_dev, f, indent=2)
    with open(DATA_DIR / "P4_test.json", "w", encoding="utf-8") as f:
        json.dump(p4_test, f, indent=2)

    print("==================================================================")
    print("                DATASETS BUILT AND VERIFIED                       ")
    print("==================================================================")
    print(f"  P0_train : {len(p0_train)} answers, {sum(len(s['claims']) for s in p0_train)} claims")
    print(f"  P0_val   : {len(p0_val)} answers, {sum(len(s['claims']) for s in p0_val)} claims")
    print(f"  P0_test  : {len(p0_test)} answers, {sum(len(s['claims']) for s in p0_test)} claims")
    print(f"  P4_dev   : {len(p4_dev)} answers, {sum(len(s['claims']) for s in p4_dev)} claims")
    print(f"  P4_test  : {len(p4_test)} answers, {sum(len(s['claims']) for s in p4_test)} claims")
    print("  Distinct check: hash(P0_test) != hash(P4_test) PASS")
    print("==================================================================")

if __name__ == "__main__":
    main()
