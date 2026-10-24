"""Small stdlib-only tests for saved-artifact parsing and identity checks."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("audit_saved_detector", Path(__file__).resolve().parents[1] / "audit_saved_detector.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def npy_bytes(shape, values, dtype="<f4", fortran=False):
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("latin1") + b"\n"
    code = {"<f4": "f", "<i8": "q", ">f4": "f"}[dtype]
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + struct.pack(dtype[0] + code * len(values), *values)


class SavedDetectorAuditTests(unittest.TestCase):
    def test_numeric_npy_preserves_endianness_and_rejects_invalid_layout(self):
        self.assertEqual(audit_module.read_npy(npy_bytes((1, 3), [1, -2, 3], ">f4")), ((1, 3), [1.0, -2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "layout"):
            audit_module.read_npy(npy_bytes((1,), [1], fortran=True))
        with self.assertRaisesRegex(ValueError, "body_length"):
            audit_module.read_npy(npy_bytes((2,), [1]))

    def test_nonfactual_case_does_not_claim_identical_evidence(self):
        comparison = audit_module.compare_inputs({"claim": "Sure!", "evidence": "reference"}, [{"text": "Sure!", "non_factual": True}])
        self.assertEqual(comparison["factual_claim_count"], 0)
        self.assertFalse(comparison["all_factual_evidence_text_identical"])

    def test_audit_aligns_original_rows_and_refuses_mismatched_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = {"macro_f1": 0, "cases": [{"index": 0, "id": "r", "label_id": 1, "prediction": 0, "response": {
                "contradicted_count": 0, "unknown_count": 0, "claim_count": 1,
                "sentences": [{"text": "A is false", "model_input_evidence": "A is true", "non_factual": False}],
            }}]}
            (root / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
            (root / "test.jsonl").write_text(json.dumps({"id": "r", "label_id": 1, "claim": "A is false.", "evidence": "A is true"}) + "\n", encoding="utf-8")
            (root / "calibration.json").write_text('{"temperature": 1}', encoding="utf-8")
            with zipfile.ZipFile(root / "predictions.npz", "w") as archive:
                archive.writestr("logits.npy", npy_bytes((1, 3), [3, 1, 0]))
                archive.writestr("labels.npy", npy_bytes((1,), [1], "<i8"))
            paths = [root / name for name in ("runtime.json", "predictions.npz", "test.jsonl", "calibration.json")]
            result = audit_module.audit(*paths)
            self.assertEqual(result["contradiction_cases"]["raw_pairwise_misses"], 1)
            self.assertTrue(result["cases"][0]["input_comparison"]["single_claim_equal_ignoring_terminal_period"])
            runtime["cases"][0]["id"] = "other"
            (root / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity_mismatch"):
                audit_module.audit(*paths)


if __name__ == "__main__":
    unittest.main()
