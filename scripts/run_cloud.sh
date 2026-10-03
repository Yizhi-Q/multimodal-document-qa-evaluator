#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
split="${1:-validation}"
case "$split" in
  validation) limit=20 ;;
  test) limit=100 ;;
  *) echo "Usage: bash scripts/run_cloud.sh validation|test" >&2; exit 2 ;;
esac
python -c 'import torch; assert torch.cuda.is_available(), "An NVIDIA GPU runtime is required"; print(torch.cuda.get_device_name(0))'
dataset="data/cord/$split/annotations.jsonl"
if [ ! -f "$dataset" ]; then
  python scripts/prepare_cord.py --split "$split"
fi
export MLLM_BACKEND=transformers
export MLLM_LOAD_IN_4BIT="${MLLM_LOAD_IN_4BIT:-true}"
python evaluate.py --dataset "$dataset" --limit "$limit"
echo "Download the run folder. Stop/terminate the GPU in your provider console when finished."
