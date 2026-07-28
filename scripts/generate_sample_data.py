"""Generate three synthetic documents and their labels."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
SCHEMA = {"vendor": "string", "document_number": "string", "date": "date", "currency": "code", "total_amount": "number"}
EXAMPLES = [
    ("invoice-001", "invoice_001.png", "INVOICE", "Northwind Studio", "INV-2026-0142", "2026-07-18", 198.0),
    ("receipt-001", "receipt_001.png", "RECEIPT", "Laneway Coffee", "R-88421", "2026-07-21", 12.0),
    ("order-001", "order_001.png", "PURCHASE ORDER", "Bluegum Supplies", "PO-7008", "2026-07-25", 55.0),
]


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_id, filename, title, vendor, number, date, total in EXAMPLES:
        image = Image.new("RGB", (1100, 760), "#f8f6ef")
        draw = ImageDraw.Draw(image)
        body, heading = ImageFont.load_default(size=28), ImageFont.load_default(size=48)
        draw.rounded_rectangle((55, 45, 1045, 715), radius=20, outline="#253858", width=4, fill="white")
        draw.text((105, 95), title, fill="#17324d", font=heading)
        lines = [vendor, f"Document: {number}", f"Date: {date}", f"TOTAL: AUD {total:.2f}"]
        for index, line in enumerate(lines):
            draw.text((110, 230 + index * 90), line, fill="#1f2933", font=body)
        image.save(IMAGE_DIR / filename)
        fields = {"vendor": vendor, "document_number": number, "date": date, "currency": "AUD", "total_amount": total}
        rows.append({
            "id": item_id,
            "image": f"images/{filename}",
            "question": "What is the total and who issued the document?",
            "field_schema": SCHEMA,
            "expected_fields": fields,
            "mock_response": {"document_type": title.lower(), "fields": fields, "answer": "Synthetic fixture", "evidence": [], "uncertainties": [], "confidence": 1.0},
        })
    (ROOT / "data" / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(rows)} synthetic documents")


if __name__ == "__main__":
    main()

