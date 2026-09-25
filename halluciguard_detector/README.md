# HalluciGuard Detector Agent

A reference-grounded, sentence-level hallucination detector. It accepts a draft LLM answer and evidence, labels each sentence as `SUPPORTED`, `CONTRADICTED`, or `NOT_ENOUGH_INFO`, and returns an answer-level hallucination risk.

This detector does **not** pretend that truth can be inferred from wording alone. Evidence is mandatory. `NOT_ENOUGH_INFO` means the supplied evidence is insufficient; it does not mean a claim is globally false.

## Design

- **Training data:** RAGTruth human span annotations, converted to sentence labels.
- **Model:** compact DeBERTa-v3 cross-encoder over `(evidence, sentence)`.
- **Leakage control:** train/dev partitioning is grouped by source document; the official RAGTruth test split is untouched.
- **Outputs:** per-sentence calibrated probabilities, exact character offsets, evidence snippets, aggregate risk, and a verifier-routing flag.
- **Calibration:** temperature scaling and the decision threshold are fitted only on dev predictions.

The implementation borrows the reference-conditioned checking formulation from MiniCheck and RefChecker, but contains original HalluciGuard code. RAGTruth is MIT licensed; MiniCheck is Apache-2.0 licensed. Their repositories are retained under `third_party/` for provenance.

The included `artifacts/detector-best` checkpoint is the completed one-epoch baseline described in [MODEL_CARD.md](MODEL_CARD.md). Its metrics are honest baseline results, not a claim of perfect open-world detection.

## Install and reproduce

```powershell
python -m pip install -r requirements.txt
python -m halluciguard_detector.cli prepare-data
python -m halluciguard_detector.cli train-model --epochs 3 --batch-size 8
python -m halluciguard_detector.cli evaluate-model
python -m pytest -q halluciguard_detector/tests
```

Training starts a fresh fine-tuning run from the named pretrained language-model checkpoint. It does not reuse HalluciGuard detector weights. Training a useful language encoder from random tokens would require vastly more data and compute and is intentionally not claimed here.

## Run

Test your own LLM answer directly:

```powershell
python -m halluciguard_detector.cli predict-text `
  --query "Who created Java?" `
  --answer "Java was created by Snehith in 1995." `
  --evidence "Java was designed by James Gosling at Sun Microsystems and released in 1995."
```

Use `-e` more than once to supply multiple evidence passages. The detector
cannot establish truth from the query and answer alone; evidence is required.

Run the complete Base LLM → Detector → n8n Verifier slice:

```powershell
python scripts/test_llm_detector_verifier_slice.py --query "Who created Java?" --force-verifier
```

```powershell
$env:HALLUCIGUARD_DETECTOR_MODEL = "artifacts/detector-best"
uvicorn halluciguard_detector.api:app --host 0.0.0.0 --port 8000
```

```json
POST /v1/detect
{
  "user_query": "Who created Java?",
  "draft_answer": "Java was created by Snehith in 1995. It was developed at Sun Microsystems.",
  "evidence": ["Java was designed by James Gosling at Sun Microsystems and released in 1995."]
}
```

## Operational boundary

This component is a triage detector, not the final Judge. Route `CONTRADICTED` and `NOT_ENOUGH_INFO` claims to HalluciGuard's evidence Verifier. Metrics are dataset-specific and must not be described as proof that the model "works perfectly" on open-world facts.
