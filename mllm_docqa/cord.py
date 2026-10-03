"""Map CORD v2 labels to a small, explicitly scoped receipt extraction task."""
from __future__ import annotations
import json

FIELD_PATHS = {
    "subtotal_amount": ("sub_total", "subtotal_price"),
    "tax_amount": ("sub_total", "tax_price"),
    "service_charge": ("sub_total", "service_price"),
    "total_amount": ("total", "total_price"),
    "cash_amount": ("total", "cashprice"),
    "change_amount": ("total", "changeprice"),
}
SCHEMA = {name: "string or null" for name in FIELD_PATHS}
QUESTION = (
    "Extract the printed receipt amounts: subtotal before charges, tax, service charge, "
    "final total, cash tendered, and change. Copy each amount exactly as printed, preserving "
    "decimal and thousands separators and any currency prefix printed with the amount. "
    "Use strings, not JSON numbers. Use JSON null (not the string \"null\") when not shown. "
    "Tax may be labelled Tax, PPN, or PB1. Keep an explicitly printed zero. "
    "Do not calculate missing amounts."
)


def convert_cord(ground_truth, item_id, image_path, source):
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    parsed = truth.get("gt_parse")
    if not isinstance(parsed, dict):
        raise ValueError("CORD row has no gt_parse object")
    expected = {}
    for field, (group, key) in FIELD_PATHS.items():
        group_labels = parsed.get(group, {})
        if not isinstance(group_labels, dict):
            raise ValueError(f"Ambiguous repeated label group: {group}")
        value = group_labels.get(key)
        if value is None:
            continue  # No label is not proof the field is absent in the image.
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Ambiguous/nontext CORD label: {group}.{key}")
        expected[field] = value
    if not expected:
        raise ValueError("No labelled amounts in the selected task")
    return {"id": item_id, "image": image_path, "question": QUESTION,
            "field_schema": dict(SCHEMA), "expected_fields": expected, "source": source}
