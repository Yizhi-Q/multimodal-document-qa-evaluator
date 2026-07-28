"""Field-level exact-match evaluation."""

from __future__ import annotations

import re
from typing import Any


def normalise(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return re.sub(r"\s+", " ", str(value)).strip().casefold() or None


def score_fields(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    results = {key: normalise(predicted.get(key)) == normalise(value) for key, value in expected.items()}
    correct, total = sum(results.values()), len(results)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "field_results": results,
    }

