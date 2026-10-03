# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 20. Failed: 0.
Dataset fingerprint: `c92e4c28c106979227554845f4c88d5d0b59e926ecfd5ec39bddc391b2afd800`

Field accuracy: **68.06%** (49/72).
Present-field accuracy: 68.06%.
All-fields document accuracy: 55.00%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| total_amount | 15 | 20 | 75.00% |
| cash_amount | 12 | 16 | 75.00% |
| change_amount | 13 | 17 | 76.47% |
| subtotal_amount | 6 | 11 | 54.55% |
| tax_amount | 3 | 8 | 37.50% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.

Receipts with output review flags: 5.
review-flags.csv uses predictions only. Flags do not alter values or scores, and an unflagged value is not verified correct.
