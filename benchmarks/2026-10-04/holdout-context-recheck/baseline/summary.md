# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 80. Failed: 0.
Dataset fingerprint: `c92e4c28c106979227554845f4c88d5d0b59e926ecfd5ec39bddc391b2afd800`

Field accuracy: **80.78%** (227/281).
Present-field accuracy: 80.78%.
All-fields document accuracy: 65.00%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| subtotal_amount | 44 | 56 | 78.57% |
| tax_amount | 25 | 39 | 64.10% |
| total_amount | 70 | 78 | 89.74% |
| cash_amount | 40 | 48 | 83.33% |
| change_amount | 43 | 47 | 91.49% |
| service_charge | 5 | 13 | 38.46% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.

Receipts with output review flags: 31.
Additional rechecks: 0; failed rechecks: 0.
review-flags.csv uses predictions only. Flags do not alter values or scores, and an unflagged value is not verified correct.
