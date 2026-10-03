# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 20. Failed: 0.
Dataset fingerprint: `c92e4c28c106979227554845f4c88d5d0b59e926ecfd5ec39bddc391b2afd800`

Field accuracy: **76.39%** (55/72).
Present-field accuracy: 76.39%.
All-fields document accuracy: 50.00%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| total_amount | 15 | 20 | 75.00% |
| cash_amount | 12 | 16 | 75.00% |
| change_amount | 15 | 17 | 88.24% |
| subtotal_amount | 8 | 11 | 72.73% |
| tax_amount | 5 | 8 | 62.50% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.

Receipts with output review flags: 9.
Additional rechecks: 2; failed rechecks: 0.
review-flags.csv uses predictions only. Flags do not alter values or scores, and an unflagged value is not verified correct.
