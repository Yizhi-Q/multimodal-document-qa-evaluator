"""Compare completed live runs with identical data and selections."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(paths):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(runs) < 2:
        raise ValueError("At least two runs are required")
    for run in runs:
        if run.get("mode") != "live" or not run.get("complete"):
            raise ValueError("Only completed live runs can be compared as model benchmarks")
        if run["dataset"]["fingerprint"] != runs[0]["dataset"]["fingerprint"]:
            raise ValueError("Different dataset fingerprints")
        if run["dataset"]["selected_ids"] != runs[0]["dataset"]["selected_ids"]:
            raise ValueError("Different document selections")
        if run.get("report_version") != runs[0].get("report_version"):
            raise ValueError("Different report versions")
        if run["runtime"]["source_sha256"] != runs[0]["runtime"]["source_sha256"]:
            raise ValueError("Different evaluation code; re-run with the same version")
    lines = ["# Model comparison", "", "Same data, document selection, and evaluator source.", "",
             "| Run | Model | 4-bit | Receipt profile | Recheck | Max pixels | Field accuracy | Document accuracy | Failed | Mean wall seconds |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for index, run in enumerate(runs, 1):
        config = run["configuration"]
        elapsed = run["performance"]["successful_examples_mean_wall_seconds"]
        timing = f"{elapsed:.3f}" if elapsed is not None else "n/a"
        lines.append(f"| {index} | {config.get('model', 'unknown')} | {config.get('load_in_4bit', 'n/a')} | "
                     f"{config.get('receipt_profile', 'baseline')} | {config.get('receipt_recheck', 'none')} | {config.get('max_pixels', 'n/a')} | "
                     f"{run['field_accuracy']:.2%} | {run['document_accuracy']:.2%} | "
                     f"{run['failed_examples']} | {timing} |")
    for index, candidate in enumerate(runs[1:], 2):
        counts = paired_changes(runs[0], candidate)
        lines += ["", f"## Paired changes: run 1 → run {index}", "",
                  "| Field | Previously wrong, now correct | Previously correct, now wrong |",
                  "|---|---:|---:|"]
        for name, bucket in counts.items():
            lines.append(f"| {name} | {bucket['improved']} | {bucket['regressed']} |")
    lines += ["", "Latency includes preprocessing, generation and parsing; model loading is excluded.",
              "The first inference is included. Check GPU, pixel/token limits and package versions before interpreting speed.", ""]
    return "\n".join(lines)


def paired_changes(baseline, candidate):
    def indexed(run):
        records = {row["id"]: row for row in run["results"]}
        selected = run["dataset"]["selected_ids"]
        if len(records) != len(run["results"]) or set(records) != set(selected):
            raise ValueError("Run results do not match the recorded document selection")
        return records

    first, second = indexed(baseline), indexed(candidate)
    if set(first) != set(second):
        raise ValueError("Different document selections")
    counts = {}
    for item_id in baseline["dataset"]["selected_ids"]:
        before = first[item_id]["score"]["field_results"]
        after = second[item_id]["score"]["field_results"]
        if set(before) != set(after):
            raise ValueError("Different scored fields")
        for name in before:
            bucket = counts.setdefault(name, {"improved": 0, "regressed": 0})
            bucket["improved"] += int(not before[name] and after[name])
            bucket["regressed"] += int(before[name] and not after[name])
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Paths to run.json")
    args = parser.parse_args()
    try:
        print(compare(args.runs))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Cannot compare: {exc}\n")
