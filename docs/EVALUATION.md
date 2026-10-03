# Evaluation protocol

## Task

Extract six amount fields from CORD images: subtotal, tax, service charge, total, cash tendered, and change. Only the image, question and field schema reach the model.

CORD v2 has official train/validation/test splits of 800/100/100 documents. We do not train here. Begin with 20 validation documents, inspect errors, freeze prompt/model/pixel limits/evaluator, then evaluate the test split. Exclusions such as ambiguous label groups are recorded in manifest.json and must be disclosed.

This subset extraction task is **not the official full CORD score**. Receipt performance does not establish steel-industry performance.

## Metrics

- Field accuracy: correct labelled fields / all labelled fields, including inference failures.
- Present-field accuracy: restricted to nonempty ground truth.
- Document accuracy: fraction with every scored field correct and no request failure.
- Per-field accuracy and support: show the number of labels scored.
- Wall time: preprocessing, generation and parsing; excludes model initialisation. First inference is included. Mock/replay times are not model speeds.
- Qwen predictions also record synchronised GPU generation time and peak allocated GPU memory on device 0. This is not total machine VRAM utilisation.

Unlabelled fields are not automatically absent. Do not claim hallucination rates from CORD fields without exhaustive negative labels.

## Matching

Text: Unicode NFKC, whitespace collapse and case-insensitive exact match. No fuzzy matching, arbitrary punctuation deletion or leading-zero stripping.

Explicit numbers: decimal equality with a decimal point and optional comma-grouped thousands. Invalid grouping, booleans, nonfinite numbers and objects are rejected. Currency prefixes are not silently removed. Dates currently match as text.

CORD amounts stay text because receipts use different separator conventions. A changed separator counts as a mismatch.

## Error analysis

errors.csv records missing_value, unexpected_value (only against explicit null), invalid_value, value_mismatch, or inference_error.

Inspect the image and prediction before assigning a human cause: ocr_reading, wrong_field, formatting, unsupported_value, label_issue, or uncertain. A wrong answer alone does not establish its cause. Make improvements using validation data only.

## Reproducibility

Keep the data manifest and attribution. Reports record annotation/image/evaluator hashes, selected IDs, package versions and model settings. Use recorded data/model revisions to repeat experiments.

A useful later experiment is 3B inference with and without 4-bit quantisation on the same GPU. Vary one factor at a time. Do not select the best test result after repeated test-set tuning.

Disconnects leave flushed predictions while run.json remains incomplete. Incomplete runs are not final benchmarks. Review reports before publishing if processing private documents.
