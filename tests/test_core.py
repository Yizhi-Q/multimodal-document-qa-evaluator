import unittest

from mllm_docqa.core import OutputParseError, parse_model_output
from mllm_docqa.scoring import score_fields


class CoreTests(unittest.TestCase):
    def test_fenced_json(self):
        result = parse_model_output('```json\n{"fields":{"total":12},"confidence":1.2}\n```', ["total", "vendor"])
        self.assertEqual(result["fields"], {"total": 12, "vendor": None})
        self.assertEqual(result["confidence"], 1.0)

    def test_invalid_response(self):
        with self.assertRaises(OutputParseError):
            parse_model_output("no json")

    def test_flat_schema_json(self):
        result = parse_model_output(
            '{"vendor":"Northwind Studio","total_amount":198.0}',
            ["vendor", "total_amount"],
        )
        self.assertEqual(
            result["fields"],
            {"vendor": "Northwind Studio", "total_amount": 198.0},
        )

    def test_field_scoring(self):
        score = score_fields({"vendor": " Northwind  Studio ", "total": 12}, {"vendor": "northwind studio", "total": 12.0})
        self.assertEqual(score["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
