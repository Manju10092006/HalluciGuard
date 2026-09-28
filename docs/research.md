# Research, datasets and model provenance

Only references identifiable from the current repository are listed. HalluciGuard's orchestration, contracts, failure policy, relation safeguards and integration code are project implementations; using a model or conceptual formulation does not imply endorsement by its authors.

| Work | Authors / year | URL | Used in HalluciGuard | Project adaptation |
|---|---|---|---|---|
| RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models | Niu et al., 2023 | https://arxiv.org/abs/2310.17379 | Human span annotations for Detector data | Converted response spans to sentence-level three-class examples; source-grouped train/dev split; official test split preserved |
| DeBERTa: Decoding-enhanced BERT with Disentangled Attention | He et al., 2020 | https://arxiv.org/abs/2006.03654 | Encoder family for trained Detector and external NLI model | Fine-tuned `microsoft/deberta-v3-xsmall` as a three-class `(evidence, sentence)` classifier |
| MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents | Tang et al., 2024 | https://arxiv.org/abs/2404.10774 | Reference-conditioned checking formulation noted by Detector documentation | No MiniCheck code is claimed as HalluciGuard implementation; project uses its own Detector code |
| RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models | Hu et al., 2024 | https://arxiv.org/abs/2310.05097 | Conceptual reference for evidence-conditioned checking | Project-specific sentence contracts, calibration and orchestration |
| BGE / FlagEmbedding reranker family | Beijing Academy of Artificial Intelligence, repository/model documentation | https://github.com/FlagOpen/FlagEmbedding | `BAAI/bge-reranker-large` for relevance reranking | Wrapped with execution diagnostics, relevance gates and fail-soft/fail-closed policies |
| Sentence-Transformers NLI cross-encoder | Model repository metadata; authors/year not documented in this repository | https://huggingface.co/cross-encoder/nli-deberta-v3-base | Three-way entailment/contradiction/neutral scoring | Combined with relation checks, source weighting, evidence semantics and four-state aggregation |
| Sentence Transformers MiniLM embedding model | Model repository metadata; authors/year not documented in this repository | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Memory vector embeddings | Used through the Memory Agent's FAISS-backed vector store |
| LangGraph | LangChain project; reference details not documented in repository | https://github.com/langchain-ai/langgraph | Stateful conditional orchestration | Project-specific nodes, bounded loops, contracts, trace and terminal routing |

## Dataset and license notes

The Detector README states that RAGTruth is MIT licensed. Model and API use remains subject to each upstream license and service terms. The repository does not establish publication-quality evaluation for every supported domain.

## What was built independently

- canonical inter-agent Pydantic contracts;
- two-phase Detector integration and evidence bridge;
- claim gating and fallback handling;
- multi-provider generation router;
- n8n normalization boundary;
- relation/entity safeguards and four-state aggregation;
- Judge precedence and bounded workflow decisions;
- evidence-authorized Corrector targeting and validation;
- ReVerifier topicality and fail-closed routing;
- acceptance-gated memory persistence and trace reporting.
