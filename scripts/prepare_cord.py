"""Download an official CORD split at a recorded revision and export receipt labels."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mllm_docqa.cord import convert_cord

REPOSITORY = "naver-clova-ix/cord-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, help="First N source rows, including any reported exclusions")
    parser.add_argument("--revision", default="main", help="Resolved to an immutable commit before downloading")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true", help="Use cached parquet at an explicit immutable revision")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.offline and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--offline requires an explicit 40-character commit SHA in --revision")
    output = args.output or Path("data/cord") / args.split
    if output.exists():
        parser.error(f"{output} already exists; reuse it or choose a new --output")
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError:
        parser.exit(2, "Install requirements-eval.txt first.\n")
    if args.offline:
        revision = args.revision
        snapshot = Path(snapshot_download(REPOSITORY, repo_type="dataset", revision=revision,
                                          local_files_only=True))
        files = sorted(str(path.relative_to(snapshot)) for path in snapshot.glob(f"data/{args.split}-*.parquet"))
    else:
        info = HfApi().dataset_info(REPOSITORY, revision=args.revision)
        revision = info.sha
        files = sorted(s.rfilename for s in info.siblings
                       if s.rfilename.startswith(f"data/{args.split}-") and s.rfilename.endswith(".parquet"))
    if not files:
        parser.exit(2, "No official split parquet file found at the requested revision.\n")
    source = {"dataset": REPOSITORY, "revision": revision, "split": args.split,
              "license": "CC-BY-4.0", "url": "https://huggingface.co/datasets/" + REPOSITORY}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows, excluded, file_hashes = [], [], {}
    source_count = 0
    for filename in files:
        local = Path(hf_hub_download(REPOSITORY, filename, repo_type="dataset", revision=revision,
                                     local_files_only=args.offline))
        digest = hashlib.sha256()
        with local.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hashes[filename] = digest.hexdigest()
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1, columns=["image", "ground_truth"]):
            if args.limit is not None and source_count >= args.limit:
                break
            record = batch.to_pylist()[0]
            item_id = f"cord-{args.split}-{source_count:04d}"
            source_count += 1
            image_path = f"images/{item_id}.png"
            try:
                row = convert_cord(record["ground_truth"], item_id, image_path, source)
            except ValueError as exc:
                excluded.append({"id": item_id, "reason": str(exc)})
                continue
            image_bytes = record["image"]["bytes"]
            if not image_bytes:
                raise ValueError("Expected image bytes embedded in official parquet")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(output / image_path)
            rows.append(row)
        if args.limit is not None and source_count >= args.limit:
            break
    if not rows:
        raise ValueError("No eligible documents; inspect the source labels")
    (output / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {**source, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_rows_examined": source_count, "included_examples": len(rows),
                "excluded": excluded, "source_files_sha256": file_hashes,
                "task": "Six receipt amount fields, scored only where a label exists",
                "selection": "First N rows in the original official split order"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "ATTRIBUTION.md").write_text(
        "# CORD attribution\n\nCORD: A Consolidated Receipt Dataset for Post-OCR Parsing.\n"
        "Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, Hwalsuk Lee (2019).\n"
        "Source: https://github.com/clovaai/cord\n"
        "Dataset: https://huggingface.co/datasets/naver-clova-ix/cord-v2\n"
        "License: https://creativecommons.org/licenses/by/4.0/\n\n"
        "Changes: images exported as RGB PNG; six amount labels mapped into a flat schema.\n"
        "This subset task is not the official complete CORD benchmark.\n", encoding="utf-8")
    print(f"Prepared {len(rows)} receipts, excluded {len(excluded)}. Revision: {revision}")
    print(output / "annotations.jsonl")


if __name__ == "__main__":
    main()
