from typing import List, Optional
from .base_loader import BaseDatasetLoader
from .benchmark_example import BenchmarkExample


class HaluEvalLoader(BaseDatasetLoader):
    """Dataset loader for HaluEval hallucination benchmark (QA, Dialogue, Summarization, General)."""

    FALLBACK_EXAMPLES: List[BenchmarkExample] = [
        # QA Pair 1
        BenchmarkExample(
            query="Which magazine was started first Arthur's Magazine or First for Women?",
            response="Arthur's Magazine",
            expected_risk="LOW",
            category="halueval_qa",
            context="Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia.",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="QA_000001",
            metadata={"original_label": "right_answer", "halueval_type": "right_answer"}
        ),
        BenchmarkExample(
            query="Which magazine was started first Arthur's Magazine or First for Women?",
            response="First for Women was started first.",
            expected_risk="HIGH",
            category="halueval_qa",
            context="Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia.",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="QA_000001",
            metadata={"original_label": "hallucinated_answer", "halueval_type": "hallucinated_answer"}
        ),
        # Dialogue Pair 1
        BenchmarkExample(
            query="Do you like Iron Man?",
            response="I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well.",
            expected_risk="LOW",
            category="halueval_dialogue",
            context="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite.",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="DIALOG_000001",
            metadata={"original_label": "right_response", "halueval_type": "right_response"}
        ),
        BenchmarkExample(
            query="Do you like Iron Man?",
            response="I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.",
            expected_risk="HIGH",
            category="halueval_dialogue",
            context="[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite.",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="DIALOG_000001",
            metadata={"original_label": "hallucinated_response", "halueval_type": "hallucinated_response"}
        ),
        # Summarization Pair 1
        BenchmarkExample(
            query="Summarize the article on Germanwings Flight 9525.",
            response="Marseille prosecutor says so far no videos were used in the crash investigation despite media reports.",
            expected_risk="LOW",
            category="halueval_summarization",
            context="Marseille, France (CNN) The French prosecutor leading an investigation into the crash of Germanwings Flight 9525...",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="SUM_000001",
            metadata={"original_label": "right_summary", "halueval_type": "right_summary"}
        ),
        BenchmarkExample(
            query="Summarize the article on Germanwings Flight 9525.",
            response="A video showing the final moments of Germanwings Flight 9525 has been recovered by investigators from the wreckage site.",
            expected_risk="HIGH",
            category="halueval_summarization",
            context="Marseille, France (CNN) The French prosecutor leading an investigation into the crash of Germanwings Flight 9525...",
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="SUM_000001",
            metadata={"original_label": "hallucinated_summary", "halueval_type": "hallucinated_summary"}
        ),
        # General Domain Sample
        BenchmarkExample(
            query="Produce a list of common words in the English language.",
            response="the, a, and, to, in, that, is, it, of, for, with, was, on, be, by, at, as, but, from, this",
            expected_risk="LOW",
            category="halueval_general",
            context=None,
            source_dataset="HaluEval",
            source_model="ChatGPT",
            pair_id="GENERAL_000001",
            metadata={"hallucination": "no", "hallucination_spans": [], "original_label": "no"}
        )
    ]

    def __init__(self, subset: str = "qa", use_hf_datasets: bool = True) -> None:
        """Initialize HaluEval loader.
        
        Args:
            subset: One of 'qa', 'dialogue', 'summarization', 'general'.
            use_hf_datasets: Whether to attempt loading via Hugging Face datasets library.
        """
        self.subset: str = subset
        self.use_hf_datasets: bool = use_hf_datasets

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """Loads HaluEval benchmark examples with preserved pairs, context, and metadata."""
        examples: List[BenchmarkExample] = []

        if self.use_hf_datasets:
            try:
                from datasets import load_dataset
                dataset = load_dataset("pminervini/HaluEval", name=self.subset, split="data", streaming=True)
                
                row_idx = 0
                prefix_map = {
                    "qa": "QA",
                    "dialogue": "DIALOG",
                    "summarization": "SUM",
                    "general": "GENERAL"
                }
                pair_prefix = prefix_map.get(self.subset, "HALU")

                for item in dataset:
                    row_idx += 1
                    pair_id = f"{pair_prefix}_{row_idx:06d}"

                    if self.subset == "qa":
                        ctx = item.get("knowledge", "")
                        q = item.get("question", "")
                        right_ans = item.get("right_answer", "")
                        halluc_ans = item.get("hallucinated_answer", "")

                        if q and right_ans:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=right_ans,
                                expected_risk="LOW",
                                category="halueval_qa",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "right_answer", "halueval_type": "right_answer"}
                            ))

                        if q and halluc_ans:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=halluc_ans,
                                expected_risk="HIGH",
                                category="halueval_qa",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "hallucinated_answer", "halueval_type": "hallucinated_answer"}
                            ))

                    elif self.subset == "dialogue":
                        ctx = item.get("dialogue_history", "") or item.get("knowledge", "")
                        q = "Respond to the dialogue turn."
                        right_resp = item.get("right_response", "")
                        halluc_resp = item.get("hallucinated_response", "")

                        if right_resp:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=right_resp,
                                expected_risk="LOW",
                                category="halueval_dialogue",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "right_response", "halueval_type": "right_response"}
                            ))

                        if halluc_resp:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=halluc_resp,
                                expected_risk="HIGH",
                                category="halueval_dialogue",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "hallucinated_response", "halueval_type": "hallucinated_response"}
                            ))

                    elif self.subset == "summarization":
                        ctx = item.get("document", "")
                        q = "Summarize the document."
                        right_sum = item.get("right_summary", "")
                        halluc_sum = item.get("hallucinated_summary", "")

                        if right_sum:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=right_sum,
                                expected_risk="LOW",
                                category="halueval_summarization",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "right_summary", "halueval_type": "right_summary"}
                            ))

                        if halluc_sum:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=halluc_sum,
                                expected_risk="HIGH",
                                category="halueval_summarization",
                                context=ctx,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={"original_label": "hallucinated_summary", "halueval_type": "hallucinated_summary"}
                            ))

                    elif self.subset == "general":
                        q = item.get("user_query", "")
                        resp = item.get("chatgpt_response", "")
                        halluc_flag = str(item.get("hallucination", "no")).lower()
                        spans = item.get("hallucination_spans", [])
                        risk = "HIGH" if halluc_flag in ["yes", "true", "1"] else "LOW"

                        if q and resp:
                            examples.append(BenchmarkExample(
                                query=q,
                                response=resp,
                                expected_risk=risk,
                                category="halueval_general",
                                context=None,
                                source_dataset="HaluEval",
                                source_model="ChatGPT",
                                pair_id=pair_id,
                                metadata={
                                    "original_label": halluc_flag,
                                    "hallucination": halluc_flag,
                                    "hallucination_spans": spans
                                }
                            ))

                    if limit and len(examples) >= limit:
                        break

            except Exception as e:
                print(f"[HaluEvalLoader] Warning: HF Datasets load unavailable ({e}). Using standardized fallback samples.")
                examples = self.FALLBACK_EXAMPLES

        if not examples:
            examples = self.FALLBACK_EXAMPLES

        if limit:
            examples = examples[:limit]

        return examples
