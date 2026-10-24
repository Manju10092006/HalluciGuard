"""Bake four pinned models for the general balanced path and Memory only."""
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    models = json.loads(Path("/app/models-pinned.json").read_text())
    for model in models:
        weight = "model.safetensors" if model["safetensors_available"] else "pytorch_model.bin"
        patterns = ["*.json", "*.model", "vocab.txt", "merges.txt", "1_Pooling/*", "2_Normalize/*", weight]
        downloaded = Path(snapshot_download(model["repo_id"], revision=model["revision"], allow_patterns=patterns))
        if not (downloaded / weight).is_file():
            raise RuntimeError("Pinned weights missing: " + model["repo_id"])
        reference = downloaded.parent.parent / "refs" / "main"
        reference.parent.mkdir(exist_ok=True)
        reference.write_text(model["revision"])
        print("Pinned model baked:", model["repo_id"], model["revision"], flush=True)


if __name__ == "__main__":
    main()
