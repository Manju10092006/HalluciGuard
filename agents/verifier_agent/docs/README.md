# HalluciGuard Verifier Agent v2.0

## Description
The Verifier Agent is the core evidence retrieval and claim verification engine for the HalluciGuard trust layer. It decomposes complex claims, fetches authoritative evidence from various domain-specific sources, and validates facts using NLI (Natural Language Inference) models.

## Architecture Overview
The system implements a robust 8-stage pipeline:
1. **Domain Validation**: Ensures the claim is routed to the correct context.
2. **Claim Decomposition**: Breaks compound claims into atomic sub-claims.
3. **Query Expansion**: Enriches the claim with domain terminology.
4. **Multi-source Retrieval**: Queries specialized adapters (e.g., PubMed, arXiv).
5. **Aggregation + Dedup**: Consolidates results.
6. **Hybrid Retrieval**: Dense (semantic) + Sparse (BM25) filtering.
7. **Cross-encoder Reranking**: Re-scores top passages against the claim.
8. **NLI Entailment**: Predicts logical entailment, contradicts, or neutral.
*(Additional sub-stages include Evidence Scoring, Conflict Resolution, Explanation Generation, and Citation Formatting)*

## Quick Start Guide

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Copy `.env.example` to `.env` and fill in credentials.
2. Configure thresholds in `config/credibility.yaml` and `config/retry.yaml`.

### Run the Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /verify` : Takes `VerifierInputV2` and returns a structured `VerifierOutputV2`.
- `GET /health` : Returns system health including adapter statuses and cache metrics.
- `GET /domains` : Lists domain statistics, credibility scores, and implementation status.
- `GET /pipeline` : Returns metadata describing the pipeline stages.
- `GET /metrics` : Returns the latest `MetricsCollector` summary.

## Domain Adapter Coverage
| Domain | Status | Sources |
|---|---|---|
| Healthcare | Fully Implemented | PubMed, ClinicalTrials |
| Cybersecurity | Fully Implemented | NVD, MITRE |
| AI Research | Fully Implemented | arXiv |
| Finance | Stub | - |
| General | Fully Implemented | Wikipedia |
*(Total: 5 fully implemented, 18 stubs)*

## Testing Guide
Run the test suite using pytest:
```bash
pytest tests/ -v
```

## Benchmark Guide
To run standard benchmarks (PubHealth, FEVER):
```bash
python -m benchmarks.runner
```

## Output Contract v2 JSON Example
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "domain": "healthcare",
  "reports": [
    {
      "claim_id": "claim_1",
      "claim_text": "Vitamin C cures cancer",
      "verdict": "likely_hallucinated",
      "explanation": "Extensive trials show no curative effect...",
      "evidence": []
    }
  ],
  "metrics": {}
}
```
