# Evaluation result

Live model inference on the specified dataset

Mode: live. Examples: 20. Failed: 2.
Dataset fingerprint: `903fa9394f6c048dde281ccd65bf66fb31fd5692dc6f3ed8a1209e568f9781be`

Field accuracy: **63.89%** (46/72).
Present-field accuracy: 63.89%.
All-fields document accuracy: 50.00%.

| Field | Correct | Scored | Accuracy |
|---|---:|---:|---:|
| total_amount | 13 | 20 | 65.00% |
| cash_amount | 11 | 16 | 68.75% |
| change_amount | 13 | 17 | 76.47% |
| subtotal_amount | 7 | 11 | 63.64% |
| tax_amount | 2 | 8 | 25.00% |

Unlabelled fields are not scored. Failed requests count as incorrect fields.
Error categories describe output differences, not proven OCR or reasoning causes.
Inspect images and fill review_category/review_notes in errors.csv to establish causes.
