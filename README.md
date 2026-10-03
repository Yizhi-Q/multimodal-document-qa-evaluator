# Multimodal Document QA Evaluator

English | [简体中文](README.zh-CN.md)

A personal project for extracting structured fields from document images and evaluating the results. It supports an OpenAI-compatible API and local or cloud NVIDIA inference with Qwen2.5-VL.

The current version includes official CORD receipt import, schema-aware scoring, per-document predictions, error review, and comparable run reports. A real GPU evaluation was completed on 2026-10-03 using Qwen2.5-VL-3B-Instruct and an RTX 5090. The three bundled synthetic documents remain pipeline fixtures.

## Measured results

Both settings used the same 95 eligible CORD v2 test receipts and 319 labelled amount fields. Of the original 100 test receipts, four contained ambiguous or nontext amount labels and one had no labelled amounts for this task. Computation used BF16 in both runs.

| Weights | Correct fields | Field accuracy | All-fields document accuracy | Failed receipts | Mean wall seconds | Maximum recorded allocated GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| NF4, 4-bit | 268/319 | 84.01% | 72.63% | 0 | 2.284 | 2.458 |
| BF16, unquantised | 272/319 | 85.27% | 72.63% | 0 | 1.595 | 7.204 |

NF4 used 65.9% less recorded allocated memory and took longer per receipt on this GPU. Timing includes preprocessing, generation and parsing, excluding model loading. Memory is PyTorch allocated memory, not total machine VRAM. This is a six-field extraction task, not the official full CORD benchmark or an industrial deployment result.

[Full benchmark and raw records](docs/BENCHMARK.md) · [Five validation examples reviewed](docs/VALIDATION_REVIEW.md)

## Start here

- [Cloud GPU run guide in Chinese](docs/CLOUD_RUN.md): RunPod, AutoDL, or another Linux NVIDIA machine.
- [Tested AutoDL setup](docs/AUTODL_RUN.md): model verification, BF16 configuration and offline caches.
- [Evaluation protocol](docs/EVALUATION.md): dataset splits, metrics, comparisons and limitations.
- [Notebook](notebooks/cloud_evaluation.ipynb): the same workflow in Jupyter.
- [Learning checklist](docs/LEARNING.md): what to understand and what to implement yourself.

## Offline checks

Python 3.10+. Tests and the mock pipeline require only the standard library.

~~~bash
python -m unittest discover -s tests -v
python evaluate.py --dry-run
~~~

Each run creates a new directory under artifacts/runs/. Dry runs are marked **PIPELINE CHECK ONLY**. Mock predictions cannot establish model accuracy.

## Run Qwen on a cloud GPU

Use Linux with working NVIDIA drivers and CUDA-enabled PyTorch. A single 24 GB GPU is a reasonable starting configuration for Qwen2.5-VL-3B in 4-bit mode with the supplied image limits; exact memory and throughput must be measured.

~~~bash
python -m pip install -r requirements-local.txt -r requirements-eval.txt
python -m unittest discover -s tests -v
bash scripts/run_cloud.sh validation
~~~

This downloads the official CORD v2 validation split, selects its first 20 eligible exported documents for the pilot, loads Qwen, and writes real predictions. The importer records source order and exclusions. After inspecting validation errors and freezing settings:

~~~bash
bash scripts/run_cloud.sh test
~~~

Use the test set for final evaluation; do not repeatedly tune prompts against its errors. Each official split download is a few hundred MB; weights and packages need additional storage. Downloads happen on the cloud machine.

**The script does not stop or delete a paid cloud instance.** Download results, then stop/terminate it in the provider console and check remaining storage charges.

## Data and reports

~~~text
scripts/prepare_cord.py     Official split downloader and label export
scripts/cache_verified_model.py  Verify and cache a locally downloaded model snapshot
mllm_docqa/cord.py          CORD task definition and label mapping
mllm_docqa/dataset.py       Input validation and content fingerprints
mllm_docqa/core.py          Model backends and JSON parsing
mllm_docqa/scoring.py       Schema-aware exact match
mllm_docqa/evaluation.py    Reports, metrics and review files
evaluate.py                Live / dry-run / saved-prediction scoring
compare_runs.py            Guarded comparison of matching live runs
app.py                     Interactive Gradio demo
tests/                     Offline regression tests
~~~

Each evaluation creates:

- run.json: settings, model revision when available, package versions, data/source fingerprints and metrics.
- predictions.jsonl: one flushed record per document, retained if the process is interrupted.
- errors.csv: expected/predicted values, observable mismatch categories and columns for human review.
- summary.md: readable results with the evaluation mode displayed.

Missing labels are unscored. Failed requests stay in the denominator and cause exit status 1. Existing output directories are never overwritten. Intermediate reports have complete=false.

To score without another model call:

~~~bash
python evaluate.py --dataset data/cord/validation/annotations.jsonl \
  --limit 20 --predictions artifacts/runs/YOUR_RUN/predictions.jsonl
~~~

The source run.json must sit beside predictions and match the dataset fingerprint. Replay preserves whether its source was real inference or a fixture.

Compare two real runs:

~~~bash
python compare_runs.py artifacts/runs/FIRST/run.json artifacts/runs/SECOND/run.json
~~~

Comparison rejects mock/replay runs, mismatched selections and different evaluator source versions.

## API and demo

~~~bash
python -m pip install -r requirements.txt
cp .env.example .env
~~~

Configure MLLM_BACKEND=openai, MLLM_API_KEY, MLLM_BASE_URL and MLLM_MODEL. Never commit credentials. Select a compatible Responses or Chat Completions endpoint with MLLM_API_MODE.

~~~bash
python app.py
python evaluate.py --dataset data/cord/validation/annotations.jsonl --limit 20
~~~

MLLM_BACKEND=transformers selects Qwen. MLLM_MODEL_REVISION pins a Hugging Face model commit. Pixel limits, quantisation, compute dtype and generation length are configurable in .env.example. The tested default compute dtype is bfloat16; it requires GPU BF16 support. Initial float16 validation produced invalid outputs, including without weight quantisation; those attempts are retained in the validation review.

Use requirements-benchmark.txt to reproduce the tested inference dependencies after preparing the recorded CUDA-enabled PyTorch build. Add --offline to prepare_cord.py when the official parquet files are already cached and provide an explicit immutable dataset revision.

## Scope and limitations

- This is a learning/portfolio project, not a production document service.
- CORD contains Indonesian receipts. We score six amount fields, not the full official CORD benchmark.
- Released CORD labels do not cover the old fixture's vendor, document number and date task; those labels are not fabricated.
- CORD amounts are matched as printed strings. Explicit numeric schemas support decimal points and valid comma grouping; locales are not guessed.
- Confidence and evidence are model-generated output, not calibrated probabilities or independently verified grounding.
- No training or fine-tuning is performed. Pretraining contamination of a public benchmark cannot be ruled out.
- One GPU and one complete run per test setting were measured; concurrent serving and performance on other hardware were not evaluated.

## Data attribution

CORD by Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo and Hwalsuk Lee (2019), [official project](https://github.com/clovaai/cord), [CORD v2 release](https://huggingface.co/datasets/naver-clova-ix/cord-v2). Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Exported images and labels retain separate attribution; this project's MIT license covers its code.
