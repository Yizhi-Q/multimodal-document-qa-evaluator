# Multimodal Document QA Evaluator

A small, end-to-end MLLM project for extracting structured fields from document images, answering grounded questions, and measuring field-level accuracy.

The repository is designed as a learning project and portfolio piece: it includes a browser demo, provider-compatible model calls, synthetic evaluation data, an offline dry run, and unit tests.

## What it demonstrates

- Multimodal prompting with an image and a natural-language question
- Schema-guided JSON extraction with defensive output parsing
- Evidence, uncertainty, and confidence fields for inspectable predictions
- Batch evaluation using field-level exact match
- A Gradio interface for interactive testing
- API-free fixtures for reproducible local validation

## Project structure

```text
.
├── app.py                         # Interactive Gradio demo
├── evaluate.py                    # Offline and live batch evaluation
├── mllm_docqa/
│   ├── core.py                    # API client, prompt, image encoding, parsing
│   └── scoring.py                 # Field-level evaluator
├── scripts/generate_sample_data.py
├── data/
│   ├── annotations.jsonl
│   └── images/                    # Synthetic invoices and receipts
├── tests/test_core.py
├── artifacts/                     # Generated evaluation reports
└── .env.example
```

## Quick start

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python -m unittest discover -s tests -v
python evaluate.py --dry-run
```

The dry run uses stored mock predictions, so it verifies the data and evaluation pipeline without an API key. Its score is a pipeline check, not a model benchmark.

## Run with a multimodal model

Copy the example environment file and add credentials locally. Never commit `.env`.

```bash
cp .env.example .env
```

The client supports an OpenAI-compatible endpoint. Configure these values in `.env`:

```dotenv
MLLM_API_KEY=replace_me
MLLM_BASE_URL=https://api.openai.com/v1
MLLM_MODEL=gpt-4.1-mini
MLLM_API_MODE=responses
```

Start the interface:

```bash
python app.py
```

Or evaluate the labelled sample set with live model calls:

```bash
python evaluate.py
```

Set `MLLM_API_MODE=chat` when using a compatible provider that implements Chat Completions rather than the Responses API.

## Evaluation format

Each line in `data/annotations.jsonl` contains an image path, a question, the requested field schema, and expected fields. `evaluate.py` writes a detailed report to `artifacts/evaluation_results.json`.

To turn this into a stronger portfolio project, add a real document dataset, compare two vision-language models, report latency and cost, and include error categories such as OCR failure, field ambiguity, and hallucinated values.

## Current limitations

- The bundled dataset is synthetic and intentionally small.
- Exact-match scoring does not yet handle semantic equivalence or partial credit.
- Live evaluation requires credentials for a compatible multimodal model.
- Predictions should be manually reviewed before use in consequential workflows.

## License

This learning project is provided under the MIT License.
