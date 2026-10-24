"""One CPU-only historical pair; no retrieval models, threshold fitting or training."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time


def available_memory_mb():
    if os.name != "nt":
        return None
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_page", "avail_page",
                "total_virtual", "avail_virtual", "avail_extended")]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot inspect available memory")
    return status.avail_phys / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index", type=int, default=301)
    args = parser.parse_args()
    free = available_memory_mb()
    if free is not None and free < 3072:
        print(json.dumps({"status": "blocked", "reason": "less_than_3072_MB_free", "free_mb": round(free)}))
        return 2
    # These overrides are process-local. Never use CUDA or other retrieval models.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from halluciguard_detector.detector import Detector
    free_after_import = available_memory_mb()
    if free_after_import is not None and free_after_import < 2048:
        print(json.dumps({"status": "blocked", "reason": "less_than_2048_MB_free_after_import", "free_mb": round(free_after_import)}))
        return 2
    with (args.data_dir / "test.jsonl").open(encoding="utf-8") as handle:
        row = next((json.loads(line) for i, line in enumerate(handle) if i == args.index), None)
    if row is None:
        raise ValueError("Requested row not found")
    detector = Detector(root / "artifacts" / "detector-best", device="cpu", evidence=object())
    trace = {}
    probabilities = detector._classify(row["claim"], row["evidence"], trace=trace)
    prediction = max(probabilities, key=probabilities.get).value
    output = {"status": "executed", "scope": "Actual checkpoint pair classification only; no retrieval or LangGraph execution",
              "device": str(detector.device), "threads": torch.get_num_threads(),
              "index": args.index, "id": row["id"], "gold": row["label"],
              "prediction": prediction, "quality_pass": prediction == row["label"],
              "probabilities": {label.value: value for label, value in probabilities.items()},
              "tokenization": trace, "seconds": round(time.monotonic() - started, 2)}
    print(json.dumps(output, indent=2))
    return 0 if output["quality_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
