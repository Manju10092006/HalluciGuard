"""Actual fast-tokenizer preprocessing with constructed vocabulary, no model."""
import hashlib
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from nli.robust_entailment import NLIEngine


def pipeline(mismatch=False):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3,
                                  "four": 4, "claim": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    class Pipeline:
        def __init__(self): self.tokenizer = tokenizer
        def preprocess(self, item, **params):
            encoded = self.tokenizer(item["text"], text_pair=item["text_pair"],
                                     return_tensors="pt", **params)
            if mismatch:
                encoded["input_ids"][0, 0] = 99
            return encoded
        def __call__(self, batch, **params):
            for item in batch: self.preprocess(item, **params)
            return [[{"label": "entailment", "score": .9},
                     {"label": "contradiction", "score": .05},
                     {"label": "neutral", "score": .05}] for _ in batch]
    return Pipeline()


def test_survival_offsets_are_bound_to_actual_ids_and_pair():
    engine = NLIEngine()
    engine.pipeline = pipeline()
    # Directly exercise the observed invocation with a deliberately short cap.
    engine._invoke([{"text": "one two three four", "text_pair": "claim"}],
                   truncation="only_first", max_length=3)
    trace = engine.diagnostics()["submitted_inputs"][-1]
    assert trace["survival_observability"] == "verified_token_offsets"
    assert trace["retained_character_ranges"] == {"premise": [[0, 3], [4, 7]], "hypothesis": [[0, 5]]}
    assert trace["tokenizer_truncated"] is True
    assert trace["hypothesis_sha256"] == hashlib.sha256(b"claim").hexdigest()
    assert trace["retained_pair_tokens"] == 3


def test_mismatched_offset_encoding_is_never_reported_as_survival():
    engine = NLIEngine()
    engine.pipeline = pipeline(mismatch=True)
    engine._invoke([{"text": "one two", "text_pair": "claim"}], truncation=True, max_length=3)
    trace = engine.diagnostics()["submitted_inputs"][-1]
    assert trace["survival_observability"] == "offset_encoding_mismatch"
    assert "retained_character_ranges" not in trace


def test_diagnostic_reader_cannot_mutate_the_engines_trace():
    engine = NLIEngine()
    engine.pipeline = pipeline()
    engine.batch_classify("claim", ["one two"])
    snapshot = engine.diagnostics()
    assert snapshot["requested"] is True and snapshot["total_duration_ms"] >= 0
    snapshot["submitted_inputs"].clear()
    assert engine.diagnostics()["submitted_inputs"]
