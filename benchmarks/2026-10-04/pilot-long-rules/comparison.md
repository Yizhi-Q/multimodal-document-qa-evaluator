# Model comparison

Same data, document selection, and evaluator source.

| Run | Model | 4-bit | Receipt profile | Max pixels | Field accuracy | Document accuracy | Failed | Mean wall seconds |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | Qwen/Qwen2.5-VL-3B-Instruct | True | baseline | 802816 | 75.00% | 50.00% | 0 | 2.236 |
| 2 | Qwen/Qwen2.5-VL-3B-Instruct | True | label-aware-v1 | 802816 | 72.22% | 60.00% | 0 | 6.881 |

## Paired changes: run 1 → run 2

| Field | Previously wrong, now correct | Previously correct, now wrong |
|---|---:|---:|
| total_amount | 3 | 2 |
| cash_amount | 1 | 1 |
| change_amount | 0 | 2 |
| subtotal_amount | 0 | 1 |
| tax_amount | 1 | 1 |

Latency includes preprocessing, generation and parsing; model loading is excluded.
The first inference is included. Check GPU, pixel/token limits and package versions before interpreting speed.

