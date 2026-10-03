"""Run a real model, a labelled fixture check, or re-score saved predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from mllm_docqa.dataset import load_dataset
from mllm_docqa.evaluation import evaluate_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/annotations.jsonl"))
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Mock pipeline check, never model performance")
    modes.add_argument("--predictions", type=Path, help="Re-score a previous run's predictions.jsonl")
    parser.add_argument("--limit", type=int, help="Use the first N rows; selection is recorded in the report")
    args = parser.parse_args(argv)
    try:
        rows, info = load_dataset(args.dataset)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            rows = rows[:args.limit]
        info["selected_ids"] = [row["id"] for row in rows]
        info["selected_examples"] = len(rows)
        mode = "dry-run" if args.dry_run else "replay" if args.predictions else "live"
        config, analyzer, predictions = {}, None, None
        if mode == "dry-run":
            if not all(isinstance(row.get("mock_response"), dict) for row in rows):
                raise ValueError("--dry-run requires fixtures; it cannot evaluate real CORD data")
        elif mode == "replay":
            records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
            predictions = {}
            for record in records:
                if record["id"] in predictions:
                    raise ValueError("Duplicate prediction IDs")
                predictions[record["id"]] = record
            if not set(info["selected_ids"]) <= set(predictions):
                raise ValueError("Saved predictions are missing selected dataset IDs")
            manifest = args.predictions.parent / "run.json"
            if not manifest.exists():
                raise ValueError("Replay requires the source run.json beside predictions.jsonl")
            source = json.loads(manifest.read_text(encoding="utf-8"))
            if source["dataset"]["fingerprint"] != info["fingerprint"]:
                raise ValueError("Saved predictions use a different dataset fingerprint")
            config = {"source_mode": source["mode"], "source_configuration": source["configuration"],
                      "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
        if args.output is not None and args.output.exists():
            raise ValueError("Output already exists; choose a new directory")
        if mode == "live":
            from mllm_docqa.core import DocumentAnalyzer
            analyzer = DocumentAnalyzer()
            config = analyzer.describe()
        output = args.output or Path("artifacts/runs") / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + mode)
        run = evaluate_rows(rows, args.dataset.resolve(), output, mode=mode, dataset_info=info,
                            analyzer=analyzer, predictions=predictions, configuration=config)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Evaluation could not start/finish: {exc}\n")
    print(f"\n{run['claim']}\nReport: {output / 'summary.md'}")
    print(f"Field accuracy: {run['field_accuracy']:.2%}; failed documents: {run['failed_examples']}")
    return 1 if run["failed_examples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
