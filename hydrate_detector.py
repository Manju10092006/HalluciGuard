"""Fetch only the existing, immutable production Detector during a cloud build."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

URL = ("https://media.githubusercontent.com/media/Manju10092006/HalluciGuard/"
       "82cdc6fb734f5fa31cd345a4060f60178bea2118/artifacts/detector-best/model.safetensors")
EXPECTED_SIZE = 283348948
EXPECTED_SHA256 = "470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9"


def write_verified(stream, destination: Path, size: int, digest: str) -> None:
    """Verify a bounded stream before publishing it; preserve existing files."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            checksum = hashlib.sha256()
            received = 0
            while chunk := stream.read(64 * 1024):
                received += len(chunk)
                if received > size:
                    raise ValueError("Detector download exceeded expected size")
                checksum.update(chunk)
                handle.write(chunk)
        if received != size or checksum.hexdigest() != digest:
            raise ValueError("Detector size or SHA256 mismatch")
        # A hard link publishes the already verified complete file atomically,
        # and fails if any file (including an LFS pointer) already exists.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    target = Path("/app/artifacts/detector-best/model.safetensors")
    with urllib.request.urlopen(URL, timeout=90) as response:
        write_verified(response, target, EXPECTED_SIZE, EXPECTED_SHA256)
    print("Production Detector hydrated; exact size and SHA256 verified", flush=True)


if __name__ == "__main__":
    main()
