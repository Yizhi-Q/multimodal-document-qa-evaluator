#!/usr/bin/env bash
# Controlled prompt experiment using the existing, verified AutoDL caches.
set -euo pipefail
cd "$(dirname "$0")/.."
phase="${1:-pilot}"
case "$phase" in
  pilot) offset=0; limit=20 ;;
  holdout) offset=20; limit=80 ;;
  *) echo "Usage: bash scripts/run_receipt_improvement.sh pilot|holdout" >&2; exit 2 ;;
esac
export RECEIPT_DATASET="${RECEIPT_DATASET:-data/cord/validation/annotations.jsonl}"
if [ ! -f "$RECEIPT_DATASET" ]; then
  python scripts/prepare_cord.py --split validation --offline \
    --revision 7f0115a4b758a71d6473b8d085751692da2fef98
fi
python - <<'PY'
import json
import os
from pathlib import Path
from mllm_docqa.cord import QUESTION, SCHEMA
from mllm_docqa.dataset import load_dataset

path = Path(os.environ["RECEIPT_DATASET"])
manifest = json.loads((path.parent / "manifest.json").read_text())
assert manifest["split"] == "validation", "This experiment requires validation data"
assert manifest["revision"] == "7f0115a4b758a71d6473b8d085751692da2fef98", "Unexpected dataset revision"
rows, _ = load_dataset(path)
assert len(rows) == 100, "Use the full verified validation export before selecting a subset"
assert all(row["question"] == QUESTION and row["field_schema"] == SCHEMA for row in rows), "Unexpected task definition"
print("Verified the 100-image validation export; original answers are unchanged.")
PY
export MLLM_BACKEND=transformers
export MLLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export MLLM_MODEL_REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3
export MLLM_LOAD_IN_4BIT=true
export MLLM_COMPUTE_DTYPE=bfloat16
export MLLM_MIN_PIXELS=200704
export MLLM_MAX_PIXELS=802816
export MLLM_MAX_NEW_TOKENS=512
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests -v
run_root="artifacts/runs/receipt-labels-$phase-$(date -u +%Y%m%dT%H%M%SZ)"
candidate="${RECEIPT_CANDIDATE:-label-context-v1}"
case "$candidate" in
  label-aware-v1|receipt-compact-v1|amount-recheck-v1|percent-crop-v1|label-context-v1) ;;
  *) echo "Unknown RECEIPT_CANDIDATE" >&2; exit 2 ;;
esac
for profile in baseline "$candidate"; do
  export MLLM_RECEIPT_RECHECK=none
  export MLLM_RECEIPT_PROFILE="$profile"
  if [ "$profile" = amount-recheck-v1 ]; then
    export MLLM_RECEIPT_PROFILE=baseline
    export MLLM_RECEIPT_RECHECK=invalid-amount-v1
  elif [ "$profile" = percent-crop-v1 ]; then
    export MLLM_RECEIPT_PROFILE=baseline
    export MLLM_RECEIPT_RECHECK=percent-crop-v1
  elif [ "$profile" = label-context-v1 ]; then
    export MLLM_RECEIPT_PROFILE=baseline
    export MLLM_RECEIPT_RECHECK=label-context-v1
  fi
  python evaluate.py --dataset "$RECEIPT_DATASET" --offset "$offset" --limit "$limit" \
    --output "$run_root/$profile"
done
python compare_runs.py "$run_root/baseline/run.json" "$run_root/$candidate/run.json" \
  > "$run_root/comparison.md"
echo "Comparison saved: $run_root/comparison.md"
echo "Review improvements and regressions in pilot before freezing the prompt and running holdout."
echo "Save/download results, then stop the paid instance in your provider console."
