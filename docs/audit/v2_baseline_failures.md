# HalluciGuard Verifier V2 — Baseline Failure Reproduction Audit
Date: 2026-08-22
Branch: `verifier-v1.4-certification-checkpoint`
Status: Unmodified baseline prior to V2 fixes.

This document records verbatim live executions of the 4 known diagnostic failure cases.

## FAILURE 1: FALSE VERIFICATION (NLI conflation: Telangana != India)
**Claim**: `Hyderabad is the capital of India.`
**Domain**: `general` | **Retrieval Mode**: `hybrid`

```
FINAL VERDICT : [VERIFIED]
Trust Score   : 64.7%
Confidence    : 62.2%
Support Score : 62.2%
Contradict    : 0.0%
Retrieved     : 9 passages (3 verified items)
Explanation   : Verified (64.7% trust score): 1 out of 3 sources support this claim. The primary source (Wikipedia, authority: 0.80) states: "Article [National Capital Region (India)]: The National Capital Region is a region centred on the city of Delhi, a special union territory of India that hosts the country's capi..." Published unknown.

--- CITATIONS ---
[1] [NEUTRAL] Source: wikipedia | URL: https://en.wikipedia.org/wiki/National_Capital_Region_%28India%29
    NLI Match   : 0.0% | Credibility: 80.0%
    Snippet     : "Article [National Capital Region (India)]: The National Capital Region is a region centred on the city of Delhi, a special union territory of India that hosts the country's capital city New Delhi. It encompasses the entirety of Delhi and a number of adjacent districts from the states of Haryana, Uttar Pradesh, and Rajasthan. The NCR and the associated National Capital Region Planning Board (NCRPB) were created in 1985 to plan the developme"
[2] [NEUTRAL] Source: wikipedia | URL: https://en.wikipedia.org/wiki/Hyderabad%2C_Pakistan
    NLI Match   : 0.1% | Credibility: 80.0%
    Snippet     : "Article [Hyderabad, Pakistan]: Hyderabad, also known as Neroonkot, is the capital and largest city of the Hyderabad Division in the Sindh province of Pakistan. It is the second-largest city in Sindh, after Karachi, and the 7th largest in Pakistan."
[3] [ENTAILMENT] Source: wikipedia | URL: https://en.wikipedia.org/wiki/Hyderabad
    NLI Match   : 93.6% | Credibility: 80.0%
    Snippet     : "Article [Hyderabad]: Hyderabad is the capital and largest city of the Indian state of Telangana. It occupies 650 km2 (250 sq mi) on the Deccan Plateau along the banks of the Musi River, in the northern part of South India. With an average altitude of 536 m (1,759 ft), much of Hyderabad is situated on hilly terrain around artificial lakes, including the Hussain Sagar lake, predating the city's founding, in the north of"

--- RETRIEVAL TRACE ---
{
  "requested_domain": "general",
  "primary_adapter": "general",
  "retrieval_mode": "hybrid",
  "primary": {
    "called": true,
    "result_count": 4,
    "usable_count": 4,
    "relevant_count": 4,
    "top_relevance": 0.85,
    "source_diversity": 1,
    "sufficient": true,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "latency_ms": 833,
    "error": null
  },
  "tavily": {
    "called": false,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "query": "",
    "search_depth": "advanced",
    "result_count": 0,
    "extracted_count": 0,
    "usable_count": 0,
    "latency_ms": 0,
    "error": null,
    "domains_requested": []
  },
  "merged": {
    "candidate_count": 4,
    "deduplicated_count": 4
  },
  "final": {
    "reranked_count": 4,
    "decision_grade_count": 0
  },
  "evidence_details": [],
  "timings": {}
}
```

## FAILURE 2: MISSED CONTRADICTION (NLI absence/neutral: Paris != London)
**Claim**: `The Eiffel Tower is located in London.`
**Domain**: `general` | **Retrieval Mode**: `hybrid`

```
FINAL VERDICT : [UNVERIFIED]
Trust Score   : 0.0%
Confidence    : 0.0%
Support Score : 0.0%
Contradict    : 0.0%
Retrieved     : 8 passages (3 verified items)
Explanation   : Unverified: Evaluated 3 evidence items from Wikipedia, but current evidence remains inconclusive or neutral regarding the specific claim. Note: There is a genuine conflict between credible sources.

--- CITATIONS ---
[1] [NEUTRAL] Source: wikipedia | URL: https://en.wikipedia.org/wiki/Eiffel_Tower
    NLI Match   : 0.0% | Credibility: 80.0%
    Snippet     : "Article [Eiffel Tower]: The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889."
[2] [NEUTRAL] Source: wikipedia | URL: https://en.wikipedia.org/wiki/Eiffel_Tower_%28Paris%2C_Texas%29
    NLI Match   : 0.0% | Credibility: 80.0%
    Snippet     : "Article [Eiffel Tower (Paris, Texas)]: Texas's Eiffel Tower is a landmark in the city of Paris, Texas. The tower was constructed in 1993. It is a rough scale model of the Eiffel Tower in Paris, France; at 65 ft (20 m) in height, it is roughly one-sixteenth of the height of the original. It is located adjacent to the Love Civic Center and the Red River Valley Veterans Memorial in the southeastern part of the city."
[3] [NEUTRAL] Source: tavily:www.toureiffel.paris | URL: https://www.toureiffel.paris/en/access-map
    NLI Match   : 0.0% | Credibility: 80.0%
    Snippet     : "The Eiffel Tower is located in the heart of Paris, in the 7th arrondissement, on the Champ de Mars, and is very easy to access. Its official address is: 5 avenue Anatole France, 75007 Paris, France. We recommend taking public transport to come here: metro, RER, or bus. The Eiffel Tower is very well-connected to the metro – there are three stations in the nearby area. On line 6, the Bir Hakeim station is the closest, less than 10 minutes’ walk from Entrance 1 (Allée des Refuzniks) of the monum..."

--- RETRIEVAL TRACE ---
{
  "requested_domain": "general",
  "primary_adapter": "general",
  "retrieval_mode": "hybrid",
  "primary": {
    "called": true,
    "result_count": 4,
    "usable_count": 4,
    "relevant_count": 4,
    "top_relevance": 0.85,
    "source_diversity": 1,
    "sufficient": false,
    "reason": "PRIMARY_EVIDENCE_INSUFFICIENT_TERM_COVERAGE (2/3 terms covered)",
    "latency_ms": 750,
    "error": null
  },
  "tavily": {
    "called": true,
    "reason": "PRIMARY_EVIDENCE_INSUFFICIENT_TERM_COVERAGE (2/3 terms covered)",
    "query": "the eiffel tower location",
    "search_depth": "advanced",
    "result_count": 5,
    "extracted_count": 5,
    "usable_count": 5,
    "latency_ms": 2196,
    "error": null,
    "domains_requested": []
  },
  "merged": {
    "candidate_count": 9,
    "deduplicated_count": 8
  },
  "final": {
    "reranked_count": 8,
    "decision_grade_count": 0
  },
  "evidence_details": [],
  "timings": {}
}
```

## FAILURE 3: MISSED CONTRADICTION IN HYBRID MODE (Wikipedia lead lacks family details, quality gate skips Tavily)
**Claim**: `Chiranjeevi is the father of Allu Arjun.`
**Domain**: `general` | **Retrieval Mode**: `hybrid`

```
FINAL VERDICT : [UNVERIFIED]
Trust Score   : 0.0%
Confidence    : 0.0%
Support Score : 0.0%
Contradict    : 0.0%
Retrieved     : 5 passages (0 verified items)
Explanation   : No supporting or contradicting evidence was found from any authoritative source.

--- CITATIONS ---
  No decision-grade citations passed threshold.

--- RETRIEVAL TRACE ---
{
  "requested_domain": "general",
  "primary_adapter": "general",
  "retrieval_mode": "hybrid",
  "primary": {
    "called": true,
    "result_count": 5,
    "usable_count": 5,
    "relevant_count": 5,
    "top_relevance": 0.85,
    "source_diversity": 1,
    "sufficient": true,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "latency_ms": 916,
    "error": null
  },
  "tavily": {
    "called": false,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "query": "",
    "search_depth": "advanced",
    "result_count": 0,
    "extracted_count": 0,
    "usable_count": 0,
    "latency_ms": 0,
    "error": null,
    "domains_requested": []
  },
  "merged": {
    "candidate_count": 5,
    "deduplicated_count": 5
  },
  "final": {
    "reranked_count": 5,
    "decision_grade_count": 0
  },
  "evidence_details": [],
  "timings": {}
}
```

## FAILURE 4: FALSE CONFLICT (Oak sub-article triggers 20% contradiction)
**Claim**: `Java was created by James Gosling.`
**Domain**: `general` | **Retrieval Mode**: `hybrid`

```
FINAL VERDICT : [CONFLICTED]
Trust Score   : 48.3%
Confidence    : 39.8%
Support Score : 67.2%
Contradict    : 35.4%
Retrieved     : 8 passages (2 verified items)
Explanation   : Conflicted: Available evidence shows conflicting findings (1 supporting vs 1 contradicting). A primary source (Wikipedia, authority: 0.80) states: "Article [James Gosling]: James Arthur Gosling is a Canadian computer scientist, best known as the founder and lead designer behind the Java programming language.".

--- CITATIONS ---
[1] [ENTAILMENT] Source: wikipedia | URL: https://en.wikipedia.org/wiki/James_Gosling
    NLI Match   : 99.3% | Credibility: 80.0%
    Snippet     : "Article [James Gosling]: James Arthur Gosling is a Canadian computer scientist, best known as the founder and lead designer behind the Java programming language."
[2] [CONTRADICTION] Source: wikipedia | URL: https://en.wikipedia.org/wiki/Oak_%28programming_language%29
    NLI Match   : 20.3% | Credibility: 80.0%
    Snippet     : "Article [Oak (programming language)]: Oak is a discontinued programming language created by James Gosling in 1989, initially for Sun Microsystems' set-top box project. The language later evolved to become Java."

--- RETRIEVAL TRACE ---
{
  "requested_domain": "general",
  "primary_adapter": "general",
  "retrieval_mode": "hybrid",
  "primary": {
    "called": true,
    "result_count": 5,
    "usable_count": 5,
    "relevant_count": 5,
    "top_relevance": 0.85,
    "source_diversity": 1,
    "sufficient": true,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "latency_ms": 854,
    "error": null
  },
  "tavily": {
    "called": false,
    "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
    "query": "",
    "search_depth": "advanced",
    "result_count": 0,
    "extracted_count": 0,
    "usable_count": 0,
    "latency_ms": 0,
    "error": null,
    "domains_requested": []
  },
  "merged": {
    "candidate_count": 5,
    "deduplicated_count": 5
  },
  "final": {
    "reranked_count": 5,
    "decision_grade_count": 0
  },
  "evidence_details": [],
  "timings": {}
}
```
