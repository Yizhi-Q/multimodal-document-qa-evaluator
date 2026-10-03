# Model comparison

Same data, document selection, and evaluator source.

| Run | Model | 4-bit | Receipt profile | Max pixels | Field accuracy | Document accuracy | Failed | Mean wall seconds |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | Qwen/Qwen2.5-VL-3B-Instruct | True | baseline | 802816 | 75.00% | 50.00% | 0 | 2.211 |
| 2 | Qwen/Qwen2.5-VL-3B-Instruct | True | receipt-compact-v1 | 802816 | 68.06% | 55.00% | 0 | 2.138 |

## Paired changes: run 1 → run 2

| Field | Previously wrong, now correct | Previously correct, now wrong |
|---|---:|---:|
| total_amount | 3 | 3 |
| cash_amount | 2 | 2 |
| change_amount | 0 | 2 |
| subtotal_amount | 1 | 3 |
| tax_amount | 1 | 2 |

Latency includes preprocessing, generation and parsing; model loading is excluded.
The first inference is included. Check GPU, pixel/token limits and package versions before interpreting speed.

