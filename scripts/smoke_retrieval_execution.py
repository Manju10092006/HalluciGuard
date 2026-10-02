"""Real cached-model retrieval smoke on cited, fixed passages; not live search."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("smoke artifact already exists")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "verifier_agent"))
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    from retrievers.hybrid import HybridRetriever
    from rerankers.cross_encoder import CrossEncoderReranker
    from schemas.models import Passage
    passages = [Passage(title="Hanoi Investment Map", source="government", source_id="hanoi-government",
        url="https://hanoiinvestment.hanoi.gov.vn/en", publication_date="unknown",
        snippet="Ha Noi is the capital of the Socialist Republic of Vietnam and the national political and administrative center."),
        Passage(title="Microsoft history", source="microsoft", source_id="microsoft-history",
        url="https://www.microsoft.com/en-us/about", publication_date="unknown",
        snippet="Microsoft was founded by Bill Gates and Paul Allen in 1975.")]
    hybrid = HybridRetriever()
    selected = hybrid.retrieve("What is the capital of Vietnam?", passages, k=2, dense_model=args.dense_model)
    reranker = CrossEncoderReranker()
    ranked = reranker.rerank("Hanoi is the capital of Vietnam.", selected, k=1)
    report = {"input_origin": "fixed cited passages, not newly retrieved web evidence",
              "backend_execution": hybrid.diagnostics(), "reranker_execution": reranker.diagnostics(),
              "selected": [{"source_id": p.source_id, "url": p.url, "snippet": p.snippet} for p in ranked]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
