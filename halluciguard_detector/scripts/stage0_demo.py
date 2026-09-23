"""
Stage 0 demonstration script.
Runs claim extraction on 10 sample answers.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from halluciguard_detector.detector import StandaloneDetector

sample_answers = [
    ("Who created Java?", "Java was created by James Gosling at Sun Microsystems in 1995. It was designed to be platform independent."),
    ("What is Python?", "Python is a high-level interpreted programming language created by Guido van Rossum and released in 1991."),
    ("Who invented the light bulb?", "Thomas Edison invented the practical incandescent light bulb in 1879 after testing thousands of materials."),
    ("What is the capital of France?", "The capital of France is Paris. It is famous for the Eiffel Tower and the Louvre museum."),
    ("Who wrote the theory of relativity?", "Albert Einstein published the special theory of relativity in 1905 and general relativity in 1915."),
    ("When did WWII end?", "World War II ended in 1945 following the surrender of Germany in May and Japan in September."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("Who founded Microsoft?", "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for the Altair 8800."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa in the early 16th century during the Italian Renaissance.")
]

def main():
    detector = StandaloneDetector()
    print("==================================================================")
    print("           STAGE 0 REPORT: SAMPLE CLAIM EXTRACTION (10 ANSWERS)   ")
    print("==================================================================")
    for idx, (q, a) in enumerate(sample_answers, 1):
        resp = detector.detect(user_query=q, draft_answer=a, request_id=f"DEMO-{idx:03d}")
        print(f"\n[{idx}] Query: {q}")
        print(f"    Answer: {a}")
        print(f"    Status: {resp.status.value} | Mode: {resp.extraction_mode.value} | Claims Extracted: {len(resp.claims)}")
        for c in resp.claims:
            print(f"      - {c.claim_id}: {c.text}")

if __name__ == "__main__":
    main()
