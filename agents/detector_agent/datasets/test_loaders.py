"""Verification test script demonstrating research-grade dataset representation and backward compatibility."""

from detector_agent.datasets import BenchmarkExample, HaluEvalLoader, TruthfulQALoader


def main():
    print("=========================================================")
    print(" HalluciGuard - Upgraded Research Dataset Verification")
    print("=========================================================\n")

    # 1. Test HaluEval Loader
    halueval = HaluEvalLoader(subset="qa")
    halueval_samples = halueval.load_dataset(limit=4)

    print(f"--- 1. Loaded {len(halueval_samples)} Research-Grade HaluEval Samples ---\n")
    for i, ex in enumerate(halueval_samples, start=1):
        assert isinstance(ex, BenchmarkExample)
        print(f"Sample {i}: [{ex.category}] Pair ID: {ex.pair_id} | Expected Risk: {ex.expected_risk}")
        print(f"  Source Dataset: {ex.source_dataset} | Source Model: {ex.source_model}")
        print(f"  Query:          '{ex.query}'")
        print(f"  Response:       '{ex.response}'")
        print(f"  Context:        '{ex.context}'")
        print(f"  Metadata:       {ex.metadata}")
        
        # Verify backward compatibility dictionary export
        dict_export = ex.to_dict()
        assert "query" in dict_export and "response" in dict_export and "expected_risk" in dict_export
        assert "context" in dict_export and "pair_id" in dict_export
        print(f"  [OK] to_dict() Export Validated.\n")

    # 2. Test HaluEval General Config (with Hallucination Spans)
    halueval_gen = HaluEvalLoader(subset="general")
    gen_samples = halueval_gen.load_dataset(limit=2)

    print(f"--- 2. Loaded {len(gen_samples)} HaluEval General Samples ---\n")
    for i, ex in enumerate(gen_samples, start=1):
        assert isinstance(ex, BenchmarkExample)
        print(f"Sample {i}: [{ex.category}] Pair ID: {ex.pair_id} | Expected Risk: {ex.expected_risk}")
        print(f"  Query:          '{ex.query}'")
        print(f"  Response:       '{ex.response}'")
        print(f"  Original Label: {ex.metadata.get('original_label')}")
        print(f"  Halluc Spans:   {ex.metadata.get('hallucination_spans')}\n")

    # 3. Test TruthfulQA Loader
    truthfulqa = TruthfulQALoader(subset="generation")
    truthfulqa_samples = truthfulqa.load_dataset(limit=4)

    print(f"--- 3. Loaded {len(truthfulqa_samples)} Research-Grade TruthfulQA Samples ---\n")
    for i, ex in enumerate(truthfulqa_samples, start=1):
        assert isinstance(ex, BenchmarkExample)
        print(f"Sample {i}: [{ex.category}] Pair ID: {ex.pair_id} | Expected Risk: {ex.expected_risk}")
        print(f"  Source Dataset: {ex.source_dataset} | Source Model: {ex.source_model}")
        print(f"  Query:          '{ex.query}'")
        print(f"  Response:       '{ex.response}'")
        print(f"  Metadata:       {ex.metadata}\n")

    print("=========================================================")
    print(" All Upgraded Dataset Infrastructure Tests Passed!")
    print("=========================================================")


if __name__ == "__main__":
    main()
