import os
import json

dir_path = r'C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets'
fname = 'halluciguard_dataset.jsonl'
path = os.path.join(dir_path, fname)

# Hardcoded syntheses for the 49 hallucinated records in Stage 1
synthesized_map = {
    0: "Arthur's Magazine was published in the 19th century, but the evidence does not state when First for Women was started.",
    1: "The Oberoi Group has its head office in Delhi.",
    2: "Matt Groening named the character after President Richard Nixon's middle name.",
    4: "James Henry Miller's wife was American.",
    6: "The evidence states that Jonathan Stark won two Grand Slam titles, but does not mention Henri Leconte.",
    8: "The Indogrammodes genus of moth contains only one species.",
    10: "Badr Hari was once considered the best kickboxer in the world, but has been involved in controversies regarding unsportsmanlike conduct and crimes of violence.",
    12: "The Mount Panorama Circuit track is 6.213 km long.",
    13: "The guest appearance is from El-P.",
    15: "The narrator of \"Frontier\" was Walter Coy, who starred in Gunmen from Laredo.",
    17: "The form of music played by Die Rhöner Säuwäntzt, Skiffle, originated in the United States in the first half of the 20th century.",
    19: "Malcolm Smith was named Most Valuable Player of Super Bowl XLVIII.",
    21: "U.S. Highway 60 is also known as the historic Midland Trail.",
    23: "Annette Bening received a star on the Hollywood Walk of Fame in 2006.",
    24: "The current members of Metallica are James Hetfield, Lars Ulrich, Kirk Hammett, and Robert Trujillo.",
    25: "House aired on the Fox network.",
    27: "The eponymous debut studio album was released in 2017.",
    29: "Catherine Cortez Masto previously served as the 32nd Attorney General of Nevada.",
    31: "Longs Drugs stores are located throughout the state of Hawaii.",
    33: "Donahue replaced Kelli Ward.",
    35: "The Wolfhounds were formed in 1985, and Hole was formed in 1989.",
    37: "The female main protagonist, Katniss Everdeen, is 16 years old.",
    39: "Chang Ucchin was born during Japanese colonial rule, which ended at the conclusion of World War II.",
    41: "Old School is directed by Todd Phillips.",
    43: "New Faces of 1952 helped jump start the career of American actress Carol Lawrence.",
    45: "Pavel Urysohn and Leonid Levin were not known for the same type of work; Urysohn was a mathematician and Levin is a computer scientist.",
    47: "Kings of Leon is an American rock band, but The New Pornographers is a Canadian indie rock band.",
    48: "Both 750 Seventh Avenue and 101 Park Avenue are located in New York City.",
    50: "Kimberly Ann Hart was played by actress Amy Jo Johnson.",
    52: "Aleksander Ford was born first in 1908.",
    54: "Yes, both Jane and First for Women are women's magazines.",
    55: "Both Nicholas Ray and Elia Kazan were American film directors.",
    56: "Polaris is based in Roseau, Minnesota, USA.",
    58: "The Saimaa Gesture is a documentary about Finnish rock groups.",
    60: "David Lee Roth was inducted into the Rock and Roll Hall of Fame in 2007.",
    62: "The sister school is located in Nassau County, New York.",
    64: "McClellan Air Force Base was located in California.",
    66: "Junkers was based in Dessau, Germany.",
    67: "The University of Providence is a private Roman Catholic university.",
    69: "The song 'I Still Haven't Found What I'm Looking For' is from the album 'The Joshua Tree'.",
    71: "The album was recorded in New Paltz, which is a village in Ulster County, New York.",
    73: "Tammy Wynette was born in May of 1942 and sang the duet.",
    75: "Francis Nethersole was born first in 1587.",
    77: "The evidence mentions the Reinheitsgebot (German Beer Purity Law), which limits the ingredients in beer in Germany.",
    79: "The other member is Dennis Howard Marks.",
    81: "Robert Sheehan starred in The Messenger.",
    83: "The paloma is a tequila-based cocktail, but a gin and tonic is made with gin.",
    85: "Glenn Hughes is older, born in 1951.",
    88: "The Semmering railway was built with a standard gauge track."
}

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

os.makedirs('training/data', exist_ok=True)
out_file = 'training/data/shape_b_llm_synthesized_stage1.jsonl'

shape_b_records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        record = json.loads(line)
        target = record.get('target', {})
        if not isinstance(target, dict): continue
        if 'actions' in target and 'analysis' in target: continue # skip Shape A
        shape_b_records.append(record)

stage1 = shape_b_records[:90]

with open(out_file, 'w', encoding='utf-8') as out:
    for i, r in enumerate(stage1):
        verdict = r.get('input',{}).get('judge',{}).get('overall_verdict','').lower()
        prompt = build_prompt(r)
        if verdict == 'hallucinated':
            target_resp = synthesized_map.get(i, "")
        else:
            target_resp = r.get('input',{}).get('llm_response', '')
            
        out.write(json.dumps({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target_resp}
            ]
        }) + '\n')

print(f"Processed 90 Stage 1 records and wrote to {out_file}")
