# Bugfix Plan: Corrector Agent Disclaimer Override

## 1. Trace the Pipeline End-to-End
For the test case provided, the pipeline executes as follows:
1. **Planner**: Evaluates the `HALLUCINATED` claim. Because `evidenceIds` are present, it places the claim in `claimsToRewrite` (not `unsupportedClaims`).
2. **Model Client**: The fine-tuned Qwen model is invoked. However, because of a catastrophic data flaw (detailed in section 3), it generates either the disclaimer or garbage.
3. **Merger (Diff Generation)**: The `merger.computeDiffs` function deterministically builds the `diffs` array based on the *Planner's* output, NOT the model's output. It sees the claim in `claimsToRewrite`, extracts the evidence passage ("Python was created by Guido..."), and marks it `REWRITTEN_WITH_EVIDENCE`.
4. **Judge Verification**: The Judge evaluates the model's actual raw output. Because the model failed to incorporate the required evidence, the Judge rejects it. This loop repeats until `maxRetries` is hit, leaving `is_approved = False`.
5. **Orchestrator Divergence**: In `app/orchestrator.py` (lines 148-154), the code explicitly checks `if is_approved:`. Because the model was rejected, it falls into the `else:` block where `final_text` is **hardcoded** to `"A fully verified response could not be produced with the available evidence."`

**Conclusion**: The disclaimer text is **(a) hardcoded in code** (in `orchestrator.py`) and triggered by the boolean `is_approved == False` check, which fires because the LLM failed the Judge's verification loop.

## 2. Check Trust Score / Approval Threshold Logic
There is **no** `trust_score < threshold` boundary causing the discard. Instead, the orchestrator completely ignores the successfully resolved `diffs` and relies entirely on the boolean `is_approved` flag from the Judge. If the LLM generates a bad response (which it does, due to the broken model), the orchestrator falls back to the hardcoded generic disclaimer. The bug is architectural: the pipeline shouldn't discard perfectly good deterministic diffs just because the LLM generation step failed.

## 3. Dataset Audit & Discrepancy (Answers to Step 1)
1. **Exact Files Currently Read**: 
   `training/build_dataset.py` currently reads from three files in `C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets`:
   - `halluciguard_dataset.jsonl` (187 records)
   - `halluciguard_ragtruth.jsonl` (84 records)
   - `halluciguard_truthfulqa.jsonl` (498 records)
2. **Leftover Files Confirmation**:
   Yes, the three files currently in `C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets` are the old leftover files. They use a nested `input`/`target` schema, not the exact flat schema provided.
3. **Discrepancy Explanation**:
   The record count came out to 769 (187 + 84 + 498) because the script was pointing to the old leftover directory (`C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets`) rather than the actual new dataset. Because my code incorrectly assumed the *new* flat schema (`r.get('query')`, etc.), it failed to extract any fields from the nested `input/target` schema of the leftover files, resulting in 0 evidence for all 769 records. I will fix `build_dataset.py` to point to `<DATASETS_FOLDER_PATH>` and map the fields using the exact flat schema you provided.

## 4. Orchestrator Role Definition & Architecture Fix (Answer to Step 2)
**Role Definition Chosen:** (b) The fine-tuned model is only used for phrasing/fluency polish on top of evidence text the merger already assembled.

**Data Flow & Fix Implementation**:
Since the deterministic merger can successfully extract exact evidence passages (producing a highly accurate but potentially less fluent response), the fine-tuned model's job is simply to take the evidence and preserved claims and weave them into a polished final text. 
- If the model generates a response that passes the Judge (`is_approved == True`), we use the model's fluent output as the `final_text`.
- If the model fails the Judge (`is_approved == False`), we **do not discard the work**. Instead, we gracefully fall back to stitching the deterministic `correctedText` from the `diffs` array (which contains the raw evidence passages). 
- The generic disclaimer will *only* be used for claims that are genuinely `UNSUPPORTED` per-claim (which `merger.py` already handles by inserting the disclaimer into that specific claim's `correctedText`).

**File to Edit**: `app/orchestrator.py`
In `executeCorrectionPipeline`, instead of the generic disclaimer fallback, we will do:
```python
if is_approved:
    final_text = merger.cleanAndFormatResponse(corrected_text)
else:
    # Fallback to the stitched diffs so we never lose resolved claims!
    final_text_parts = [d.correctedText for d in final_diffs]
    final_text = " ".join(final_text_parts)
```

## 5. Proposed Fix (Data)
**File**: `training/build_dataset.py`
Update the script to use `<DATASETS_FOLDER_PATH>` and the exact flat schema:
```python
DATASETS_DIR = r"<DATASETS_FOLDER_PATH>"
# ...
query = r.get("query", "")
orig_resp = r.get("original_response", "")
evidence = r.get("verified_evidence", [])
```

## 6. Test Cases
The following 5 test cases will pass after the fix is implemented:

1. **Full Evidence (The Reported Bug)**
   - *Input*: `HALLUCINATED` claim with evidence ("Guido...").
   - *Final Corrected Response*: `"Python was created by Guido van Rossum, and first released on February 20, 1991."`
2. **Zero Evidence**
   - *Input*: `UNSUPPORTED` claim with empty `evidenceIds`.
   - *Final Corrected Response*: `"Current evidence is insufficient to support this claim."`
3. **Mixed Multi-Claim (Verified + Rewritten)**
   - *Input*: Claim 1 is `VERIFIED`. Claim 2 is `HALLUCINATED` with evidence.
   - *Final Corrected Response*: `"[Text of Claim 1] [Evidence Text for Claim 2]"`
4. **Mixed Multi-Claim (Evidence + Unsupported)**
   - *Input*: Claim 1 `HALLUCINATED` with evidence. Claim 2 `UNSUPPORTED` with no evidence.
   - *Final Corrected Response*: `"[Evidence Text for Claim 1] Current evidence is insufficient to support this claim."`
5. **Fully Verified**
   - *Input*: All claims are `VERIFIED`.
   - *Final Corrected Response*: `"[Text of Claim 1] [Text of Claim 2]"` (Identical to original response, preserved exactly).
