"""
HaluEval Dataset Loader and Normalizer — v2 (root-cause fix).

Changes from v1:
  1. Removed ungrounded QA/dialogue duplicates that caused length/style bias.
  2. Added source_id to every example for sample-level group splitting.
  3. Added general_upsample_factor to amplify production-style full-sentence examples.
  4. Single canonical format_detector_input() used by both training and inference.
  5. Group-based train/val/test splitting by source_id (no data leakage).
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset

logger = logging.getLogger(__name__)

# Label constants — must match model config id2label exactly
NO_HALLUCINATION = 0
HALLUCINATION = 1


# ============================================================
# CANONICAL FORMATTER — single source of truth
# ============================================================

def format_detector_input(query: str, response: str, context: Optional[str] = None) -> str:
    """
    Canonical formatter for detector input.

    Used by BOTH training and inference — do NOT duplicate this logic elsewhere.

    Args:
        query:    User query or question.
        response: LLM response / answer to classify.
        context:  Optional reference context (e.g., knowledge base, retrieved doc).
                  If None or empty, the example is contextless (production path).

    Returns:
        Formatted input string for the classifier.
    """
    parts = []
    if query and query.strip():
        parts.append(f"Query: {query.strip()}")
    if context and context.strip():
        ctx = context.strip()
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        parts.append(f"Context: {ctx}")
    if response and response.strip():
        parts.append(f"Answer: {response.strip()}")
    return "\n".join(parts)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class HaluEvalConfig:
    """Configuration for HaluEval dataset loading."""
    max_rows_per_config: Optional[int] = None   # None = all rows
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    configs_to_load: list = field(
        default_factory=lambda: ["qa", "dialogue", "summarization", "general"]
    )
    # Upsample general config so full-sentence contextless examples are well-represented.
    # General config is the ONLY source of ungrounded full-sentence examples.
    # Default 5x makes it ~30% of the final training set.
    general_upsample_factor: int = 5


# ============================================================
# PER-CONFIG NORMALIZERS
# ============================================================

def _normalize_qa(dataset: Dataset, max_rows: Optional[int] = None) -> list[dict]:
    """
    Normalize the QA config — GROUNDED ONLY.

    *** KEY CHANGE vs v1: ungrounded (no-context) QA duplicates are REMOVED. ***

    HaluEval QA right_answer is always a short entity (1–5 words).
    HaluEval QA hallucinated_answer is always a full sentence (8–25 words).
    Adding ungrounded copies made the model learn: short = correct, sentence = hallucinated.
    This was the root cause of the production failure. Fix: keep only grounded examples
    so the model uses context to make the decision, not answer style.
    """
    examples = []
    data = dataset if max_rows is None else dataset.select(range(min(max_rows, len(dataset))))

    for row_idx, row in enumerate(data):
        source_id = f"qa_{row_idx}"
        query = row.get("question", "") or ""
        knowledge = row.get("knowledge", "") or ""
        correct_answer = row.get("right_answer", "") or ""
        hallucinated_answer = row.get("hallucinated_answer", "") or ""

        # GROUNDED only — no ungrounded copies
        if correct_answer.strip():
            examples.append({
                "text": format_detector_input(query, correct_answer, knowledge),
                "label": NO_HALLUCINATION,
                "source_id": source_id,
                "config": "qa",
                "has_context": True,
            })

        if hallucinated_answer.strip():
            examples.append({
                "text": format_detector_input(query, hallucinated_answer, knowledge),
                "label": HALLUCINATION,
                "source_id": source_id,
                "config": "qa",
                "has_context": True,
            })

    return examples


def _normalize_dialogue(dataset: Dataset, max_rows: Optional[int] = None) -> list[dict]:
    """
    Normalize the dialogue config — GROUNDED ONLY.

    *** KEY CHANGE vs v1: ungrounded (no-context) dialogue duplicates are REMOVED. ***

    Same reasoning as QA: the dialogue history and knowledge provide context.
    Ungrounded copies encoded style bias (responses vs knowledge snippets).
    """
    examples = []
    data = dataset if max_rows is None else dataset.select(range(min(max_rows, len(dataset))))

    for row_idx, row in enumerate(data):
        source_id = f"dialogue_{row_idx}"
        knowledge = row.get("knowledge", "") or ""
        dialogue_history = row.get("dialogue_history", "") or ""
        correct_response = row.get("right_response", "") or ""
        hallucinated_response = row.get("hallucinated_response", "") or ""

        context = f"{dialogue_history}\n{knowledge}".strip() if knowledge else dialogue_history
        # Extract last human utterance as the query
        if "[Human]:" in dialogue_history:
            query = dialogue_history.split("[Human]:")[-1].split("[Assistant]:")[0].strip()
        else:
            query = ""

        # GROUNDED only — no ungrounded copies
        if correct_response.strip():
            examples.append({
                "text": format_detector_input(query, correct_response, context),
                "label": NO_HALLUCINATION,
                "source_id": source_id,
                "config": "dialogue",
                "has_context": True,
            })

        if hallucinated_response.strip():
            examples.append({
                "text": format_detector_input(query, hallucinated_response, context),
                "label": HALLUCINATION,
                "source_id": source_id,
                "config": "dialogue",
                "has_context": True,
            })

    return examples


def _normalize_summarization(dataset: Dataset, max_rows: Optional[int] = None) -> list[dict]:
    """Normalize the summarization config — grounded (document as context)."""
    examples = []
    data = dataset if max_rows is None else dataset.select(range(min(max_rows, len(dataset))))

    for row_idx, row in enumerate(data):
        source_id = f"summarization_{row_idx}"
        document = row.get("document", "") or ""
        correct_summary = row.get("right_summary", "") or ""
        hallucinated_summary = row.get("hallucinated_summary", "") or ""

        if correct_summary.strip():
            examples.append({
                "text": format_detector_input(
                    "Summarize the following document.", correct_summary, document
                ),
                "label": NO_HALLUCINATION,
                "source_id": source_id,
                "config": "summarization",
                "has_context": True,
            })

        if hallucinated_summary.strip():
            examples.append({
                "text": format_detector_input(
                    "Summarize the following document.", hallucinated_summary, document
                ),
                "label": HALLUCINATION,
                "source_id": source_id,
                "config": "summarization",
                "has_context": True,
            })

    return examples


def _normalize_general(
    dataset: Dataset,
    max_rows: Optional[int] = None,
    upsample_factor: int = 1,
    seed: int = 42,
) -> list[dict]:
    """
    Normalize the general config.

    This is the ONLY source of ungrounded full-sentence examples.
    Both correct and hallucinated labels use full ChatGPT-style responses,
    so there is no answer-style correlation with the label.

    upsample_factor: repeat each example N times (with jitter-free exact repetition)
    to boost its representation in the training set.
    """
    examples = []
    data = dataset if max_rows is None else dataset.select(range(min(max_rows, len(dataset))))

    for row_idx, row in enumerate(data):
        source_id = f"general_{row_idx}"
        user_query = row.get("user_query", "") or ""
        chatgpt_response = row.get("chatgpt_response", "") or ""
        hallucination = row.get("hallucination", "")

        # Label from official hallucination field
        if isinstance(hallucination, str):
            label = HALLUCINATION if hallucination.strip().lower() == "yes" else NO_HALLUCINATION
        elif isinstance(hallucination, (int, float)):
            label = HALLUCINATION if int(hallucination) == 1 else NO_HALLUCINATION
        else:
            continue

        text = format_detector_input(user_query, chatgpt_response)  # no context — ungrounded
        if not text.strip():
            continue

        # Add original + upsample copies
        for copy_idx in range(upsample_factor):
            examples.append({
                "text": text,
                "label": label,
                "source_id": source_id,  # all copies share same source_id
                "config": "general",
                "has_context": False,
            })

    return examples


# ============================================================
# DATASET LOADING + SPLITTING
# ============================================================

_NORMALIZERS = {
    "qa": _normalize_qa,
    "dialogue": _normalize_dialogue,
    "summarization": _normalize_summarization,
}


def load_halueval(config: Optional[HaluEvalConfig] = None) -> DatasetDict:
    """
    Load, normalize, and split the HaluEval dataset.

    Splitting is done at the SOURCE_ID (original HaluEval row) level,
    guaranteeing no data leakage between train/val/test.

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
    """
    if config is None:
        config = HaluEvalConfig()

    all_examples = []

    for cfg_name in config.configs_to_load:
        print(f"[HaluEval] Loading config: {cfg_name}...")
        try:
            raw = load_dataset("pminervini/HaluEval", cfg_name, split="data")
        except Exception:
            try:
                raw = load_dataset("pminervini/HaluEval", cfg_name, split="train")
            except Exception as e:
                logger.warning(f"Failed to load config '{cfg_name}': {e}")
                print(f"[HaluEval] WARNING: Skipping '{cfg_name}': {e}")
                continue

        if cfg_name == "general":
            normalized = _normalize_general(
                raw,
                max_rows=config.max_rows_per_config,
                upsample_factor=config.general_upsample_factor,
                seed=config.seed,
            )
        else:
            normalizer = _NORMALIZERS.get(cfg_name)
            if normalizer is None:
                continue
            normalized = normalizer(raw, config.max_rows_per_config)

        n0 = sum(1 for e in normalized if e["label"] == 0)
        n1 = sum(1 for e in normalized if e["label"] == 1)
        print(f"[HaluEval] Config '{cfg_name}': {len(normalized)} examples "
              f"(label 0: {n0}, label 1: {n1})")
        all_examples.extend(normalized)

    if not all_examples:
        raise ValueError("No examples loaded from HaluEval.")

    # --- Sample-level group splitting ---
    # Collect unique source_ids, shuffle, then split
    source_ids = list({e["source_id"] for e in all_examples})
    rng = random.Random(config.seed)
    rng.shuffle(source_ids)

    n_total = len(source_ids)
    n_val   = max(1, int(n_total * config.val_ratio))
    n_test  = max(1, int(n_total * config.test_ratio))
    n_train = n_total - n_val - n_test

    train_ids = set(source_ids[:n_train])
    val_ids   = set(source_ids[n_train:n_train + n_val])
    test_ids  = set(source_ids[n_train + n_val:])

    # Verify no overlap
    assert len(train_ids & val_ids)  == 0, "LEAKAGE: train ∩ val"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train ∩ test"
    assert len(val_ids   & test_ids) == 0, "LEAKAGE: val ∩ test"

    train_ex = [e for e in all_examples if e["source_id"] in train_ids]
    val_ex   = [e for e in all_examples if e["source_id"] in val_ids]
    test_ex  = [e for e in all_examples if e["source_id"] in test_ids]

    # Drop source_id and has_context from the training columns (keep text + label)
    def keep_core(examples):
        return [{"text": e["text"], "label": e["label"]} for e in examples]

    def to_dataset(examples):
        ds = Dataset.from_list(keep_core(examples))
        return ds.cast_column("label", ClassLabel(names=["NO_HALLUCINATION", "HALLUCINATION"]))

    splits = DatasetDict({
        "train":      to_dataset(train_ex),
        "validation": to_dataset(val_ex),
        "test":       to_dataset(test_ex),
    })

    total_n0 = sum(1 for e in all_examples if e["label"] == 0)
    total_n1 = sum(1 for e in all_examples if e["label"] == 1)
    print(f"\n[HaluEval] Total examples: {len(all_examples)}")
    print(f"[HaluEval] Label distribution: 0={total_n0} ({100*total_n0/len(all_examples):.1f}%), "
          f"1={total_n1} ({100*total_n1/len(all_examples):.1f}%)")
    print(f"[HaluEval] Source IDs: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"[HaluEval] Examples:   train={len(splits['train'])}, "
          f"val={len(splits['validation'])}, test={len(splits['test'])}")
    print(f"[HaluEval] Source-level leakage: NONE (verified)")

    return splits


if __name__ == "__main__":
    import sys
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cfg = HaluEvalConfig(max_rows_per_config=max_rows)
    splits = load_halueval(cfg)
    print("\n--- Sample from training set ---")
    for i in range(min(3, len(splits["train"]))):
        ex = splits["train"][i]
        print(f"Label: {ex['label']}")
        print(f"Text:  {ex['text'][:200]}...")
        print("---")
