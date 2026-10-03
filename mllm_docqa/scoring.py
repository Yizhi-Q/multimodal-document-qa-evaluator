"""Conservative, schema-aware field scoring without fuzzy matching."""

from __future__ import annotations

import re
import math
import unicodedata
from decimal import Decimal
from typing import Any


def normalise(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite numeric field")
        return value
    if not isinstance(value, str):
        raise ValueError("Expected a scalar field, not an object or array")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold() or None


def _typed(value: Any, rule: Any) -> Any:
    value = normalise(value)
    if value is None:
        return None
    field_type = rule.get("type", "string") if isinstance(rule, dict) else str(rule or "string")
    if isinstance(value, bool):
        raise ValueError("Boolean is not a text or numeric field")
    if field_type.split(" or ")[0] in {"number", "integer"}:
        text = str(value)
        # Decimal point and correctly grouped thousands commas only.
        # Do not guess whether e.g. 1.234 means 1234 or 1.234.
        if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text):
            raise ValueError("Invalid numeric field")
        number = Decimal(text.replace(",", ""))
        if field_type.split(" or ")[0] == "integer" and number != number.to_integral_value():
            raise ValueError("Fractional value for integer field")
        return number
    if rule is not None and not isinstance(value, str):
        raise ValueError("Expected text according to the field schema")
    # Preserve numeric equivalence for legacy datasets without a schema.
    return value if isinstance(value, (int, float)) else str(value)


def score_fields(predicted: dict[str, Any], expected: dict[str, Any],
                 schema: dict[str, Any] | None = None) -> dict[str, Any]:
    details = {}
    for key, truth in expected.items():
        target = _typed(truth, (schema or {}).get(key))
        value = predicted.get(key)
        try:
            actual = _typed(value, (schema or {}).get(key))
            same = actual == target
            category = ("correct" if same else "missing_value" if actual is None
                        else "unexpected_value" if target is None else "value_mismatch")
        except ValueError:
            same, category = False, "invalid_value"
        details[key] = {"correct": same, "category": category,
                        "expected": truth, "predicted": value,
                        "expected_present": target is not None}
    results = {key: value["correct"] for key, value in details.items()}
    correct, total = sum(results.values()), len(results)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "field_results": results,
        "details": details,
    }
