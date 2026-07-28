"""Batch evaluation CLI with an API-free dry-run."""

import argparse
import json
from pathlib import Path

from mllm_docqa.scoring import score_fields


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation_results.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analyzer = None
    if not args.dry_run:
        from mllm_docqa.core import DocumentAnalyzer
        analyzer = DocumentAnalyzer()
    results = []
    for row in load_rows(args.dataset):
        prediction = row["mock_response"] if args.dry_run else analyzer.analyze(
            args.dataset.parent / row["image"], row["question"], row["field_schema"]
        )
        score = score_fields(prediction["fields"], row["expected_fields"])
        results.append({"id": row["id"], "score": score, "prediction": prediction})
    correct = sum(item["score"]["correct"] for item in results)
    total = sum(item["score"]["total"] for item in results)
    report = {
        "mode": "dry-run" if args.dry_run else "live",
        "examples": len(results),
        "field_accuracy": correct / total if total else 0.0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(results)} examples | field accuracy: {report['field_accuracy']:.1%}")


if __name__ == "__main__":
    main()

