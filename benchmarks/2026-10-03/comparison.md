# Model comparison

Same data, document selection, and evaluator source.

| Run | Model | 4-bit | Field accuracy | Document accuracy | Failed | Mean wall seconds |
|---|---|---|---:|---:|---:|---:|
| 1 | Qwen/Qwen2.5-VL-3B-Instruct | True | 84.01% | 72.63% | 0 | 2.284 |
| 2 | Qwen/Qwen2.5-VL-3B-Instruct | False | 85.27% | 72.63% | 0 | 1.595 |

Latency includes preprocessing, generation and parsing; model loading is excluded.
The first inference is included. Check GPU, pixel/token limits and package versions before interpreting speed.
