# Verifier Agent — Build Specification v2

Supersedes v1. Same role, same place in the pipeline (between Detector and Judge). This
version adds: a plug-in domain-adapter architecture, a verified source list per domain,
credibility weighting, query expansion, multi-source retrieval, caching, observability,
richer output, and an evaluation plan.

## Role (unchanged)

Given a suspicious claim and a domain, retrieve real evidence and score whether it supports
or contradicts the claim. Does not decide accept/reject — that's the Judge's job.

## Architecture: plug-in domain adapters

Every domain implements the same interface, registered in a central registry. Adding a domain
later means writing one adapter and registering it — no changes to retrieval, reranking, NLI,
or scoring code.

```python
class DomainAdapter(Protocol):
    name: str
    def search(self, query: str, k: int) -> list[Passage]: ...
    def credibility_of(self, source_id: str) -> float: ...

REGISTRY: dict[str, DomainAdapter] = {}

def register(adapter: DomainAdapter) -> None:
    REGISTRY[adapter.name] = adapter
```

## Domain scope for this milestone

Build these five fully (they cover your project's stated targets and have genuinely good free
APIs):

`healthcare` · `cybersecurity` · `finance` · `legal_general` · `ai_research`

Register empty stub adapters (return `[]`, log "not yet implemented") for the rest so the
registry and routing code already support them:

`programming` · `scientific` · `education` · `government` · `news` · `mathematics` ·
`physics` · `chemistry` · `biology` · `space` · `history` · `geography` · `economics` ·
`climate` · `sports` · `business` · `manufacturing` · `pharmaceuticals`

This is the honest version of "20 domains": the plug-in interface makes the promise
structurally true on day one, and you fill in adapters as time allows, rather than claiming
20 fully-verified domains that don't exist yet.

## Verified source list, by domain

Access tiers used below: **FREE-API** (call it directly, no key) · **FREE-API (key)** (free
tier, needs a registered key) · **STATS-ONLY** (real API, but numeric indicators, not
narrative text — useful for statistical claims, not prose claims) · **SCRAPE-OK** (openly
licensed content, no formal API, safe to index) · **SCRAPE-CAUTION** (no official API, check
ToS/robots.txt before automating) · **PAYWALLED** (metadata/abstract only, not usable as full
evidence text) · **AVOID** (defunct or unreliable).

**Healthcare**
| Source | Access | Entry point |
|---|---|---|
| PubMed / PMC (NLM/NIH) | FREE-API | `eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`, `efetch.fcgi` |
| ClinicalTrials.gov | FREE-API | official v2 REST API, JSON, no key |
| openFDA | FREE-API (key) | `api.fda.gov` — drug labels, adverse events, recalls |
| CDC | FREE-API | `data.cdc.gov` (Socrata/SODA API) |
| MedlinePlus Connect | FREE-API | health-topic lookups by code |
| WHO Global Health Observatory | STATS-ONLY | `ghoapi.azureedge.net/api` — indicators, not prose |

Use PubMed/PMC + openFDA + CDC as your primary evidence-text sources. WHO's API is real but
returns numbers ("X% prevalence"), so it's only useful when the claim itself is a statistic.

**Cybersecurity**
| Source | Access | Entry point |
|---|---|---|
| MITRE ATT&CK | FREE-API | STIX 2.1 JSON on GitHub, no auth: `github.com/mitre-attack/attack-stix-data` |
| NVD (CVEs) | FREE-API (key optional) | `services.nvd.nist.gov/rest/json/cves/2.0` |
| CISA KEV catalog | FREE-API | JSON/CSV feed, no auth, `cisa.gov/known-exploited-vulnerabilities-catalog` |
| OWASP | SCRAPE-OK | CC-licensed docs (Top 10, Cheat Sheets) |
| SANS / ISC | SCRAPE-CAUTION | mostly paywalled training; ISC has a narrow free feed |

**Finance**
| Source | Access | Entry point |
|---|---|---|
| SEC EDGAR | FREE-API | full-text search: `efts.sec.gov/LATEST/search-index?q=...` |
| World Bank | FREE-API | `api.worldbank.org/v2` — no key |
| IMF | FREE-API | SDMX-based data API |
| OECD | FREE-API | no key required |
| Alpha Vantage | FREE-API (key) | market data, generous free tier |
| Yahoo Finance | AVOID as primary | no official public API; unofficial scraping libraries can break without notice — use Alpha Vantage instead |

**Legal** — weakest domain for open APIs, be upfront about this with your team
| Source | Access | Entry point |
|---|---|---|
| India Code | SCRAPE-CAUTION | no documented public API; statute PDFs only |
| eCourts / NJDG | SCRAPE-CAUTION | case-status APIs exist, not full judgment text search |
| Government Gazette | SCRAPE-CAUTION | PDF-only, no API |

Recommendation: for this milestone, treat Legal as a small **curated** adapter — manually
collect and index a limited, clearly-licensed set of Acts rather than promising a live
scraper against sites with unclear ToS. Say this plainly to your professor; it's a real and
common constraint in legal-tech, not a shortcut you're hiding.

**AI / CS research**
| Source | Access | Entry point |
|---|---|---|
| arXiv | FREE-API | `export.arxiv.org/api/query` |
| Semantic Scholar | FREE-API (key optional) | `api.semanticscholar.org` |
| Crossref | FREE-API | metadata + abstracts, polite-pool email recommended |
| Papers with Code | **AVOID — defunct** | shut down by Meta in July 2025; do not build against it |
| IEEE Xplore, ACM DL, ScienceDirect, Nature/Springer | PAYWALLED | citation/metadata only via Crossref; not usable as full evidence text |

For "Programming" docs (MDN, language docs) later: these are scrape-and-locally-index
targets, not APIs — fine under their open licenses (MDN is CC-BY-SA), but that's a stub for
now, not this milestone.

**General**
| Source | Access | Entry point |
|---|---|---|
| Wikipedia | FREE-API | REST API |
| Wikidata | FREE-API | SPARQL/REST |

## Credibility weights — config-driven, not hardcoded

```yaml
# config/credibility.yaml
healthcare:
  openfda: 0.98
  pubmed: 0.97
  cdc: 0.96
  who_gho: 0.95
cybersecurity:
  mitre_attack: 0.97
  nvd: 0.96
  cisa_kev: 0.96
  owasp: 0.85
finance:
  sec_edgar: 0.97
  world_bank: 0.95
  alpha_vantage: 0.85
general:
  wikipedia: 0.80
```
Tune these later against your evaluation set (Section: Evaluation) rather than treating them
as fixed truths — they're a starting prior, not a law of nature.

## Pipeline v2

1. **Domain validation** — cross-check the Detector's `domain` tag with a lightweight
   zero-shot classifier over the claim text (`facebook/bart-large-mnli` works for this too,
   reusing a model you already need). If they disagree, log it and prefer the classifier's
   domain, since the Verifier is the last chance to catch a Detector mislabel.
2. **Query expansion** — expand the claim with domain terminology and known abbreviations
   before retrieval (e.g. a small domain-specific synonym table per adapter — this doesn't
   need a model, a curated JSON file per domain is enough for this milestone).
3. **Multi-source retrieval** — for the resolved domain, query every registered source
   adapter for that domain in parallel (not just one), dense (embeddings + FAISS) and sparse
   (BM25) per source.
4. **Aggregation + dedup** — merge all sources' results, drop near-duplicate passages.
5. **Cross-encoder reranking** — `BAAI/bge-reranker-v2-m3` (or lighter fallback) against the
   claim.
6. **NLI entailment** — `microsoft/deberta-v3-base-mnli` per top-ranked passage.
7. **Evidence scoring** — combine entailment margin × source credibility × recency ×
   cross-source agreement (do independent sources agree?) into `support_score` /
   `contradiction_score`.
8. **Citation formatting** — attach title, source, URL, publication date, and a short
   snippet to every evidence item, not just a bare score.

## Caching

Use SQLite as the default (a single file, no server to run or maintain) keyed on
`(domain, normalized_query)`. Redis is a reasonable upgrade later if you're already running
Docker for other agents, but it's not worth the operational overhead for this milestone.

## Observability

Log per request: `query_id`, per-stage timing (retrieval, rerank, NLI), which source
adapters were called, which failed, and the final verdict. Plain structured JSON logging
(Python's `logging` + a `request_id`) is enough — no dedicated observability stack needed yet.

## Folder structure

```
agents/verifier_agent/
  api/            # FastAPI app, routes
  adapters/        # one file per domain adapter + the registry
  retrievers/       # dense (FAISS) + sparse (BM25)
  rerankers/
  nli/
  routers/         # domain validation / auto-classification
  scorers/         # evidence scoring, credibility weighting
  cache/
  schemas/         # pydantic models for input/output contracts
  config/          # credibility.yaml, .env, settings.py
  tests/
  utils/
  logs/
  docs/
  benchmarks/
```

## Output contract v2

```json
{
  "query_id": "q_20260726_001",
  "domain": "healthcare",
  "domain_validated": true,
  "retrieved_sources": 4,
  "verified_sources": 3,
  "claim_evidence": [
    {
      "claim_id": "c1",
      "claim_text": "XYZ drug completely cures diabetes.",
      "evidence": [
        {
          "title": "Management of type 2 diabetes: a review",
          "source": "PubMed",
          "url": "https://pubmed.ncbi.nlm.nih.gov/...",
          "publication_date": "2024-03-01",
          "snippet": "No drug currently provides a complete cure for type 2 diabetes.",
          "entailment_label": "contradiction",
          "entailment_score": 0.91,
          "credibility_score": 0.97
        }
      ],
      "support_score": 0.12,
      "contradiction_score": 0.88,
      "trust_score": 0.11,
      "verdict": "likely_hallucinated"
    }
  ],
  "overall_evidence_confidence": 0.85,
  "latency_ms": 640
}
```

## Evaluation plan (scoped, not all seven benchmarks at once)

Retrieval quality: Precision@K, Recall@K, MRR. End-to-end: latency, source API success rate.
For factuality benchmarking, pick two that map to domains you actually built —
**PubHealth** (healthcare) and **FEVER or SciFact** (general/scientific claim verification) —
rather than attempting all seven listed benchmarks. That's a defensible, completable
evaluation; running all of HaluEval/FEVER/SciFact/HotpotQA/PubHealth/MultiFC in one semester
alongside building five agents is not.

## Service layer, repo conventions, acceptance criteria, non-goals

Unchanged from v1 — see `Verifier_Agent_Build_Spec.md`. The `/verify` and `/health` FastAPI
endpoints, branch discipline, and the diabetes-drug smoke test still apply exactly as
specified there.
