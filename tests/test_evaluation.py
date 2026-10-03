import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from compare_runs import compare
from evaluate import main
from mllm_docqa.cord import convert_cord
from mllm_docqa.core import OutputParseError, parse_model_output, parse_with_raw
from mllm_docqa.dataset import load_dataset
from mllm_docqa.evaluation import evaluate_rows
from mllm_docqa.scoring import score_fields


class ScoringTests(unittest.TestCase):
    def test_numeric_strings_and_grouping(self):
        score = score_fields({"total": "1,234.50"}, {"total": 1234.5}, {"total": "number"})
        self.assertEqual(score["accuracy"], 1)

    def test_invalid_grouping_cannot_silently_change_value(self):
        score = score_fields({"total": "12,34"}, {"total": 1234}, {"total": "number"})
        self.assertEqual(score["details"]["total"]["category"], "invalid_value")

    def test_numeric_labels_cannot_be_infinity(self):
        with self.assertRaises(ValueError):
            score_fields({}, {"total": float("inf")}, {"total": "number"})

    def test_text_identifiers_keep_leading_zeroes(self):
        self.assertEqual(score_fields({"id": "0012"}, {"id": "12"})["correct"], 0)

    def test_missing_and_unexpected_values(self):
        score = score_fields({"a": None, "b": "invented"}, {"a": "known", "b": None})
        self.assertEqual(score["details"]["a"]["category"], "missing_value")
        self.assertEqual(score["details"]["b"]["category"], "unexpected_value")

    def test_booleans_do_not_equal_numeric_one(self):
        self.assertEqual(score_fields({"a": True}, {"a": 1}, {"a": "number"})["correct"], 0)

    def test_objects_are_invalid_scalars(self):
        self.assertEqual(score_fields({"a": {"value": "x"}}, {"a": "x"})["correct"], 0)

    def test_unlabelled_fields_are_unscored(self):
        score = score_fields({"a": "x", "b": "y"}, {"a": "x"}, {"a": "string", "b": "string"})
        self.assertEqual(score["total"], 1)


class ParserTests(unittest.TestCase):
    def test_parse_error_retains_raw_model_text(self):
        with self.assertRaises(OutputParseError) as caught:
            parse_with_raw("unparseable model answer", ["total"])
        self.assertEqual(caught.exception.raw_output, "unparseable model answer")

    def test_empty_unrelated_object_is_a_parse_failure(self):
        with self.assertRaises(OutputParseError):
            parse_model_output("{}", ["total"])

    def test_wrong_fields_type_is_not_silently_accepted(self):
        with self.assertRaises(OutputParseError):
            parse_model_output('{"fields":[]}', ["total"])

    def test_nonfinite_confidence_is_sanitised(self):
        output = parse_model_output('{"fields":{"a":"x"},"confidence":"NaN"}', ["a"])
        self.assertEqual(output["confidence"], 0)


class CordTests(unittest.TestCase):
    def test_only_annotated_amounts_are_scored(self):
        row = convert_cord({"gt_parse": {"total": {"total_price": "12.000"}}}, "test-0", "x.png", {})
        self.assertEqual(row["expected_fields"], {"total_amount": "12.000"})
        self.assertEqual(len(row["field_schema"]), 6)
        self.assertNotIn("mock_response", row)

    def test_ambiguous_repeated_groups_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            convert_cord({"gt_parse": {"total": [{"total_price": "12"}]}}, "0", "x.png", {})


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "image.png").write_bytes(b"fixture image contents")
        self.rows = [{"id": "first", "image": "image.png", "question": "Read total",
                      "field_schema": {"total": "number"}, "expected_fields": {"total": 12},
                      "mock_response": {"fields": {"total": 12}}}]
        self.dataset = self.root / "annotations.jsonl"
        self.write_rows()

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self):
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_duplicate_ids_are_rejected(self):
        self.rows *= 2
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "unique"):
            load_dataset(self.dataset)

    def test_image_changes_change_fingerprint(self):
        _, before = load_dataset(self.dataset)
        (self.root / "image.png").write_bytes(b"different image")
        _, after = load_dataset(self.dataset)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_external_image_paths_are_rejected(self):
        self.rows[0]["image"] = "../outside.png"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "inside"):
            load_dataset(self.dataset)

    def test_failed_documents_stay_in_denominator(self):
        self.rows += [{**self.rows[0], "id": "second"}, {**self.rows[0], "id": "third",
                                                      "expected_fields": {"total": None}}]
        self.write_rows()
        rows, info = load_dataset(self.dataset)
        model = Mock()
        model.analyze.side_effect = [RuntimeError("broken"), {"fields": {"total": 12}}, RuntimeError("broken")]
        with contextlib.redirect_stdout(io.StringIO()):
            run = evaluate_rows(rows, self.dataset, self.root / "run", mode="live", dataset_info=info, analyzer=model)
        self.assertEqual(run["failed_examples"], 2)
        self.assertAlmostEqual(run["field_accuracy"], 1 / 3)
        self.assertEqual(run["present_field_accuracy"], 0.5)
        self.assertEqual(len((self.root / "run/predictions.jsonl").read_text().splitlines()), 3)
        self.assertTrue((self.root / "run/errors.csv").exists())
        for call in model.analyze.call_args_list:
            self.assertEqual(len(call.args), 3)
            self.assertEqual(call.args[2], {"total": "number"})

    def test_dry_run_is_labelled_and_cannot_overwrite(self):
        output = self.root / "check"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--dataset", str(self.dataset), "--dry-run", "--output", str(output)]), 0)
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(run["mode"], "dry-run")
        self.assertIn("PIPELINE CHECK", run["claim"])
        original = (output / "run.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--dataset", str(self.dataset), "--dry-run", "--output", str(output)])
        self.assertEqual(original, (output / "run.json").read_bytes())

    def test_replay_keeps_fixture_provenance(self):
        output = self.root / "original"
        replay = self.root / "replay"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--dataset", str(self.dataset), "--dry-run", "--output", str(output)])
            main(["--dataset", str(self.dataset), "--predictions", str(output / "predictions.jsonl"), "--output", str(replay)])
        report = json.loads((replay / "run.json").read_text())
        self.assertEqual(report["configuration"]["source_mode"], "dry-run")
        with self.assertRaisesRegex(ValueError, "live"):
            compare([output / "run.json", replay / "run.json"])

    def test_real_dataset_cannot_produce_mock_results(self):
        self.rows[0].pop("mock_response")
        self.write_rows()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--dataset", str(self.dataset), "--dry-run"])


if __name__ == "__main__":
    unittest.main()
