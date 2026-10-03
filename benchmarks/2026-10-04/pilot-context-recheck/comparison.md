# Model comparison

Same data, document selection, and evaluator source.

| Run | Model | 4-bit | Receipt profile | Recheck | Max pixels | Field accuracy | Document accuracy | Failed | Mean wall seconds |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | Qwen/Qwen2.5-VL-3B-Instruct | True | baseline | none | 802816 | 75.00% | 50.00% | 0 | 2.207 |
| 2 | Qwen/Qwen2.5-VL-3B-Instruct | True | baseline | label-context-v1 | 802816 | 76.39% | 50.00% | 0 | 3.012 |

## Paired changes: run 1 → run 2

| Field | Previously wrong, now correct | Previously correct, now wrong |
|---|---:|---:|
| total_amount | 0 | 0 |
| cash_amount | 0 | 0 |
| change_amount | 0 | 0 |
| subtotal_amount | 0 | 0 |
| tax_amount | 1 | 0 |

Latency includes preprocessing, generation and parsing; model loading is excluded.
The first inference is included. Check GPU, pixel/token limits and package versions before interpreting speed.

