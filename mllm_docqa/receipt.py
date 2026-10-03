"""Versioned receipt instructions and conservative, label-free output checks."""
from __future__ import annotations

import unicodedata
import copy
import time
import hashlib
import tempfile
from pathlib import Path


RECEIPT_PROFILES = ("baseline", "label-aware-v1", "receipt-compact-v1")
COMPACT_SYSTEM_PROMPT = """Read the receipt using only visible evidence.
Return one JSON object with this shape:
{"fields": {"requested_field": "printed amount or null"}}
Use every requested field name. Amounts are strings copied exactly from the image;
keep digits, separators and currency prefixes. Missing or unreadable amounts are
JSON null. Copy printed zeroes. Do not calculate amounts. Keep the response to
the fields object.
"""
AMOUNT_FIELDS = {
    "subtotal_amount", "tax_amount", "service_charge", "total_amount",
    "cash_amount", "change_amount",
}
FIELD_INSTRUCTIONS = {
    "subtotal_amount": "The printed subtotal before tax or service charges; use the Subtotal line, not an item price or quantity.",
    "tax_amount": "The printed monetary tax amount beside Tax, PPN or PB1. A percentage such as a tax-included notice is a rate, never a tax amount. If only a rate is printed, return null; do not calculate tax.",
    "service_charge": "The printed monetary service charge. A service percentage alone is not an amount; do not calculate it.",
    "total_amount": "The printed final total payable beside Total or Grand Total. Keep it separate from cash tendered and change, even when a payment amount is larger.",
    "cash_amount": "The printed cash tendered beside Cash or Tunai. Do not copy Total into this field unless the cash line itself prints that amount.",
    "change_amount": "The printed change beside Change or Kembali. Do not derive it by subtracting total from cash.",
}


def validate_profile(profile: str) -> str:
    if profile not in RECEIPT_PROFILES:
        raise ValueError("MLLM_RECEIPT_PROFILE must be one of: " + ", ".join(RECEIPT_PROFILES))
    return profile


def receipt_instructions(field_schema: dict, profile: str) -> str:
    validate_profile(profile)
    if profile == "baseline":
        return ""
    names = [name for name in field_schema if name in AMOUNT_FIELDS]
    if not names:
        return ""
    if any(field_schema[name] != "string or null" for name in names):
        raise ValueError(f"{profile} requires receipt amounts with schema 'string or null'.")
    if profile == "receipt-compact-v1":
        lookup = {
            "subtotal_amount": "Subtotal / Net amount: subtotal before charges, not an item quantity or item price",
            "tax_amount": "Tax / PPN / PB1: tax amount, not a percentage or a tax-included notice",
            "service_charge": "Service: printed charge amount, not a percentage",
            "total_amount": "Total / Grand Total: final payable amount",
            "cash_amount": "Cash / Tunai: cash tendered, separate from total",
            "change_amount": "Change / Kembali: printed change",
        }
        return "Match labels to fields:\n" + "\n".join(f"{name}: {lookup[name]}" for name in names)
    lines = ["Receipt extraction rules (label-aware-v1):"]
    lines.extend(f"- {name}: {FIELD_INSTRUCTIONS[name]}" for name in names)
    lines.extend([
        "- First locate the relevant printed label, then read the amount associated with that label. Labels guide the field assignment; do not copy unrelated footer text.",
        "- Return each amount as a JSON string copied from the image. Preserve every digit, comma, dot, leading zero and currency prefix. Do not reformat separators, append zeroes or use arithmetic to repair a value.",
        "- Re-read the digits and separators against the image before answering. Keep a printed zero. Return JSON null when the amount is absent or unreadable.",
        "- For each non-null amount, include its visible label and amount together in evidence.text, with the matching evidence.field. Evidence is a transcription, not an explanation or a calculated value.",
    ])
    return "\n".join(lines)


def amount_review_flags(fields: dict, field_schema: dict) -> list[dict]:
    """Flag observable invalid text; never guess separators or rewrite predictions.

    Only string receipt schemas are checked. Null is allowed and is not evidence
    that the model correctly established absence. These checks do not detect
    every wrong digit or field assignment and never consult expected answers.
    """
    flags = []
    for name in field_schema:
        if name not in AMOUNT_FIELDS or field_schema[name] != "string or null":
            continue
        value = fields.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            flags.append({"field": name, "reason": "amount_must_be_string", "value": value})
            continue
        text = unicodedata.normalize("NFKC", value).strip()
        reason = ("percentage_is_not_amount" if "%" in text
                  else "placeholder_is_not_amount" if text.casefold() in {"", "null", "none", "n/a", "nan"}
                  else "amount_has_no_digits" if not any(char.isdecimal() for char in text)
                  else None)
        if reason:
            flags.append({"field": name, "reason": reason, "value": value})
    return flags


def central_receipt_box(width, height):
    """Keep the central 80% vertically, including the full image width."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    margin = height // 10
    return (0, margin, width, height - margin)


def recheck_invalid_amounts(backend, image_path, field_schema, initial, policy="invalid-amount-v1", original_question=None):
    """Re-read only flagged fields once; preserve every model response and failure."""
    result = copy.deepcopy(initial)
    flags = amount_review_flags(initial["fields"], field_schema)
    result["review_flags"] = flags
    if policy in {"percent-crop-v1", "label-context-v1"}:
        flags = [flag for flag in flags if flag["reason"] == "percentage_is_not_amount"]
    if not flags:
        return result
    names = [flag["field"] for flag in flags]
    schema = {name: field_schema[name] for name in names}
    question = (
        "Re-read only the requested monetary amounts from the receipt image. "
        "Copy each amount exactly, including currency prefixes, commas, dots and printed zeroes. "
        "Percentages are rates, not amounts. If a monetary amount is absent or unreadable, "
        "return JSON null. Do not calculate a value.\n"
        + "\n".join(f"{name}: {FIELD_INSTRUCTIONS[name]}" for name in names)
    )
    attempt = {"requested_fields": names, "question": question, "prediction": None, "error": None,
               "policy": policy, "request_started": False}
    started = time.perf_counter()
    try:
        if policy == "label-context-v1":
            if not original_question:
                raise ValueError("Context recheck requires the original extraction question")
            attempt["question"] = original_question
            attempt["requested_fields"] = list(field_schema)
            attempt["receipt_profile"] = "label-aware-v1"
            attempt["request_started"] = True
            checked = backend.analyze(image_path, original_question, field_schema, receipt_profile="label-aware-v1")
        elif policy == "percent-crop-v1":
            from PIL import Image
            with tempfile.TemporaryDirectory(prefix="receipt-recheck-") as directory:
                detail = Path(directory) / "detail.png"
                with Image.open(image_path) as image:
                    box = central_receipt_box(*image.size)
                    image.convert("RGB").crop(box).save(detail)
                    attempt["view"] = {"kind": "central-80-percent", "box": list(box),
                                       "source_size": list(image.size),
                                       "sha256": hashlib.sha256(detail.read_bytes()).hexdigest()}
                attempt["request_started"] = True
                checked = backend.analyze(detail, question, schema)
        else:
            attempt["request_started"] = True
            checked = backend.analyze(image_path, question, schema)
        attempt["prediction"] = checked
        remaining = {flag["field"] for flag in amount_review_flags(checked["fields"], schema)}
        if policy == "percent-crop-v1":
            # A crop cannot establish that an amount is absent from the full
            # receipt. Keep the original flagged output if the crop is null.
            remaining.update(name for name in names if checked["fields"].get(name) is None)
        for name in names:
            if name not in remaining:
                result["fields"][name] = checked["fields"].get(name)
        attempt["accepted_fields"] = [name for name in names if name not in remaining]
        accepted = set(attempt["accepted_fields"])
        if accepted:
            result["evidence"] = [item for item in initial.get("evidence", [])
                                  if isinstance(item, dict) and item.get("field") not in accepted]
            result["evidence"].extend(item for item in checked.get("evidence", [])
                                      if isinstance(item, dict) and item.get("field") in accepted)
            # The initial free-form answer/confidence describe a different
            # prediction. Retain them in the trace, not on the merged result.
            result["answer"] = ""
            result["confidence"] = None
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        attempt["raw_error_output"] = getattr(exc, "raw_output", None)
    attempt["wall_seconds"] = round(time.perf_counter() - started, 6)
    result["initial_prediction"] = copy.deepcopy(initial)
    result["recheck_attempt"] = attempt
    result["review_flags"] = amount_review_flags(result["fields"], field_schema)
    metadata = result.setdefault("metadata", {})
    metadata["model_calls"] = 1 + int(attempt["request_started"])
    metadata["recheck_policy"] = policy
    metadata["raw_output_scope"] = "initial_request; see recheck_attempt for the additional response"
    first = initial.get("metadata", {})
    second = (attempt["prediction"] or {}).get("metadata", {})
    # In a failed second request its GPU timing/memory is unavailable. Preserve
    # each pass and avoid presenting first-pass metrics as whole-pipeline ones.
    for key in ("latency_seconds", "peak_gpu_memory_gib"):
        values = [record.get(key) for record in (first, second)]
        metadata[key] = (round(sum(values), 3) if key == "latency_seconds" else max(values)) \
            if not attempt["error"] and all(isinstance(value, (int, float)) for value in values) else None
    return result
