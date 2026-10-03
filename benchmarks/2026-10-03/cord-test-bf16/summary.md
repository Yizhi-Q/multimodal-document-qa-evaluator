# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 95. Failed: 0.
Dataset fingerprint: `57f007b6786f34902524e1417f4a88787517906023e3eae0e60671a9d3aa6692`

Field accuracy: **85.27%** (272/319).
Present-field accuracy: 85.27%.
All-fields document accuracy: 72.63%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| subtotal_amount | 57 | 61 | 93.44% |
| tax_amount | 29 | 39 | 74.36% |
| total_amount | 81 | 91 | 89.01% |
| cash_amount | 55 | 63 | 87.30% |
| change_amount | 43 | 53 | 81.13% |
| service_charge | 7 | 12 | 58.33% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.
