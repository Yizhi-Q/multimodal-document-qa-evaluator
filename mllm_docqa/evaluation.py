"""Persistent evaluation reports and field-level error review."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .scoring import score_fields


def runtime_info(root: Path) -> dict:
    packages = {}
    for package in ("torch", "torchvision", "transformers", "bitsandbytes", "qwen-vl-utils",
                    "accelerate", "huggingface-hub", "pyarrow", "openai", "Pillow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    digest = hashlib.sha256()
    for path in sorted(root.glob("mllm_docqa/*.py")) + [root / "evaluate.py"]:
        digest.update(path.name.encode() + path.read_bytes())
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "source_sha256": digest.hexdigest()}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def save_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_rows(rows, dataset_path, output, *, mode, dataset_info,
                  analyzer=None, predictions=None, configuration=None):
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "report_version": 2, "mode": mode,
        "claim": ("PIPELINE CHECK ONLY: mock answers, not model performance" if mode == "dry-run"
                  else "Saved predictions re-scored; not a new inference run" if mode == "replay"
                  else "Live model inference on the specified dataset"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info, "configuration": configuration or {},
        "runtime": runtime_info(Path(__file__).resolve().parents[1]),
        "results": [], "complete": False,
    }
    save_json(output / "run.json", run)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            started = time.perf_counter()
            prediction, error, raw_error_output = None, None, None
            try:
                if mode == "dry-run":
                    prediction = row["mock_response"]
                elif mode == "replay":
                    record = predictions[row["id"]]
                    if record.get("error"):
                        raise ValueError("Recorded failure: " + record["error"])
                    prediction = record["prediction"]
                else:
                    # Ground-truth labels never reach the model.
                    prediction = analyzer.analyze(dataset_path.parent / row["image"],
                                                 row["question"], row["field_schema"])
                if not isinstance(prediction, dict) or not isinstance(prediction.get("fields"), dict):
                    raise ValueError("Prediction must contain a fields object")
                json.dumps(prediction, allow_nan=False)
                score = score_fields(prediction["fields"], row["expected_fields"], row["field_schema"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raw_error_output = getattr(exc, "raw_output", None)
                prediction = None
                score = score_fields({}, row["expected_fields"], row["field_schema"])
                for detail in score["details"].values():
                    detail.update(correct=False, category="inference_error")
                score.update(correct=0, accuracy=0.0,
                             field_results={key: False for key in row["expected_fields"]})
            result = {"id": row["id"], "image": row["image"], "prediction": prediction,
                      "error": error, "score": score,
                      "raw_error_output": raw_error_output,
                      "wall_seconds": round(time.perf_counter() - started, 6)}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            run["results"].append(result)
            print(f"{len(run['results'])}/{len(rows)} {row['id']}: "
                  f"{'ERROR' if error else str(score['correct']) + '/' + str(score['total'])}", flush=True)
    run.update(summarise(run["results"]))
    run["complete"] = True
    save_json(output / "run.json", run)
    write_review_files(output, run)
    return run


def summarise(results):
    by_field, categories = {}, Counter()
    correct = total = present_correct = present_total = 0
    for result in results:
        for name, detail in result["score"]["details"].items():
            bucket = by_field.setdefault(name, {"correct": 0, "total": 0})
            bucket["correct"] += int(detail["correct"])
            bucket["total"] += 1
            correct += int(detail["correct"])
            total += 1
            if detail["expected_present"]:
                present_total += 1
                present_correct += int(detail["correct"])
            if not detail["correct"]:
                categories[detail["category"]] += 1
    for bucket in by_field.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
    elapsed = [r["wall_seconds"] for r in results if not r["error"]]
    return {
        "examples": len(results), "failed_examples": sum(bool(r["error"]) for r in results),
        "review_flagged_examples": sum(bool((r.get("prediction") or {}).get("review_flags")) for r in results),
        "rechecked_examples": sum(bool((r.get("prediction") or {}).get("recheck_attempt")) for r in results),
        "failed_rechecks": sum(bool((r.get("prediction") or {}).get("recheck_attempt", {}).get("error")) for r in results),
        "correct_fields": correct, "total_fields": total,
        "field_accuracy": correct / total if total else None,
        "present_field_accuracy": present_correct / present_total if present_total else None,
        "document_accuracy": sum(r["score"]["correct"] == r["score"]["total"] and not r["error"]
                                 for r in results) / len(results) if results else None,
        "per_field": by_field, "error_categories": dict(categories),
        "performance": {"successful_examples_mean_wall_seconds": sum(elapsed) / len(elapsed) if elapsed else None},
    }


def write_review_files(output, run):
    columns = ["id", "image", "field", "category", "expected", "predicted", "error",
               "review_category", "review_notes"]
    with (output / "errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in run["results"]:
            for key, detail in row["score"]["details"].items():
                if not detail["correct"]:
                    record = {"id": row["id"], "image": row["image"], "field": key,
                              "category": detail["category"], "error": row["error"] or "",
                              "expected": json.dumps(detail["expected"], ensure_ascii=False),
                              "predicted": json.dumps(detail["predicted"], ensure_ascii=False)}
                    writer.writerow({k: "'" + str(v) if str(v).startswith(("=", "+", "-", "@")) else v
                                     for k, v in record.items()})
    with (output / "review-flags.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "field", "reason", "predicted"])
        writer.writeheader()
        for row in run["results"]:
            for flag in (row.get("prediction") or {}).get("review_flags", []):
                writer.writerow({"id": row["id"], "image": row["image"], "field": flag["field"],
                                 "reason": flag["reason"], "predicted": json.dumps(flag["value"], ensure_ascii=False)})
    percent = lambda value: "n/a" if value is None else f"{value:.2%}"
    lines = ["# Evaluation result", "", run["claim"], "",
             f"Mode: {run['mode']}. Examples: {run['examples']}. Failed: {run['failed_examples']}.",
             f"Dataset fingerprint: `{run['dataset']['fingerprint']}`", "",
             f"Field accuracy: **{percent(run['field_accuracy'])}** ({run['correct_fields']}/{run['total_fields']}).",
             f"Present-field accuracy: {percent(run['present_field_accuracy'])}.",
             f"All-fields document accuracy: {percent(run['document_accuracy'])}.", "",
             "| Field | Correct | Scored | Accuracy |", "|---|---:|---:|---:|"]
    for field, value in run["per_field"].items():
        lines.append(f"| {field} | {value['correct']} | {value['total']} | {percent(value['accuracy'])} |")
    lines += ["", "Unlabelled fields are not scored. Failed requests count as incorrect fields.",
              "Error categories describe output differences, not proven OCR or reasoning causes.",
              "Inspect images and fill review_category/review_notes in errors.csv to establish causes.", ""]
    lines += [f"Receipts with output review flags: {run['review_flagged_examples']}.",
              f"Additional rechecks: {run['rechecked_examples']}; failed rechecks: {run['failed_rechecks']}.",
              "review-flags.csv uses predictions only. Flags do not alter values or scores, and an unflagged value is not verified correct.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
