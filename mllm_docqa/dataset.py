"""Validate inputs and fingerprint both annotations and images."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .scoring import score_fields


def load_dataset(path: Path) -> tuple[list[dict], dict]:
    path = path.resolve()
    raw = path.read_bytes()
    rows, identities, image_hashes = [], set(), {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Row must be an object")
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id.strip() or item_id in identities:
                raise ValueError("IDs must be nonempty and unique")
            for key in ("image", "question"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing {key}")
            schema, labels = row.get("field_schema"), row.get("expected_fields")
            if not isinstance(schema, dict) or not isinstance(labels, dict) or not labels:
                raise ValueError("field_schema and nonempty expected_fields are required")
            if not set(labels) <= set(schema):
                raise ValueError("Every label must have a requested schema field")
            score_fields(labels, labels, schema)
            image = (path.parent / row["image"]).resolve()
            if not image.is_relative_to(path.parent):
                raise ValueError("Image must be inside the dataset directory")
            if not image.is_file():
                raise ValueError(f"Image does not exist: {row['image']}")
            image_hashes[item_id] = hashlib.sha256(image.read_bytes()).hexdigest()
            identities.add(item_id)
            rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError("Dataset is empty")
    digest = hashlib.sha256(raw + json.dumps(image_hashes, sort_keys=True).encode()).hexdigest()
    return rows, {"annotations_sha256": hashlib.sha256(raw).hexdigest(),
                  "fingerprint": digest, "examples": len(rows), "image_sha256": image_hashes}
