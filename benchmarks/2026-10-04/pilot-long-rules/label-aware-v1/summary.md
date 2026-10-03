# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 20. Failed: 0.
Dataset fingerprint: `c92e4c28c106979227554845f4c88d5d0b59e926ecfd5ec39bddc391b2afd800`

Field accuracy: **72.22%** (52/72).
Present-field accuracy: 72.22%.
All-fields document accuracy: 60.00%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| total_amount | 16 | 20 | 80.00% |
| cash_amount | 12 | 16 | 75.00% |
| change_amount | 13 | 17 | 76.47% |
| subtotal_amount | 7 | 11 | 63.64% |
| tax_amount | 4 | 8 | 50.00% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.

Receipts with output review flags: 3.
review-flags.csv uses predictions only. Flags do not alter values or scores, and an unflagged value is not verified correct.
