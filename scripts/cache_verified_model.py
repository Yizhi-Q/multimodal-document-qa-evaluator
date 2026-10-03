"""Verify a local snapshot and reuse its files in a pinned Hugging Face cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def checksum(path, expected):
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--integrity", type=Path,
                        default=Path("benchmarks/2026-10-03/model-integrity.json"))
    parser.add_argument("--cache-dir", type=Path, help="HF hub cache; defaults to HF_HOME/hub")
    args = parser.parse_args()
    info = json.loads(args.integrity.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", info["revision"]):
        parser.error("Integrity manifest must contain an immutable model revision")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", info["repository"]):
        parser.error("Invalid model repository ID")
    source = args.model_dir.resolve(strict=True)
    cache = args.cache_dir or Path(os.getenv("HF_HOME", str(Path.home()/".cache/huggingface")))/"hub"
    home = cache.resolve()/("models--"+info["repository"].replace("/", "--"))
    verified = []
    # Finish validating all files before adding cache entries.
    for entry in info["files"]:
        relative = Path(entry["file"])
        expected = entry["checksum"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest file path")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected):
            raise ValueError("Invalid manifest checksum")
        path = (source/relative).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError("Model file resolves outside the snapshot")
        if path.stat().st_size != entry["size"] or checksum(path, expected) != expected:
            raise ValueError("Checksum mismatch: "+entry["file"])
        blob = home/"blobs"/expected
        if blob.exists() and (blob.stat().st_size != entry["size"] or checksum(blob, expected) != expected):
            raise ValueError("Existing cache blob is invalid: "+entry["file"])
        verified.append((path, blob, relative))
        print("Verified", entry["file"], flush=True)
    for path, blob, relative in verified:
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            blob.unlink(missing_ok=True)
            try:
                os.link(path, blob)
            except OSError:
                blob.symlink_to(path)
        target = home/"snapshots"/info["revision"]/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            if checksum(target, blob.name) != blob.name:
                raise ValueError("Existing snapshot file is invalid: "+str(relative))
            continue
        target.symlink_to(os.path.relpath(blob, target.parent))
    print("Pinned cache ready:", info["repository"], info["revision"])
    print("Keep the local model directory if cache entries use symbolic links.")


if __name__ == "__main__":
    main()
