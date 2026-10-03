import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from compare_runs import paired_changes
from evaluate import main
from mllm_docqa.core import DocumentAnalyzer, OutputParseError, _build_prompt, parse_model_output
from mllm_docqa.receipt import amount_review_flags, central_receipt_box, recheck_invalid_amounts


SCHEMA = {name: "string or null" for name in ("tax_amount", "total_amount", "cash_amount", "change_amount")}


class ReceiptTests(unittest.TestCase):
    def test_baseline_keeps_original_prompt(self):
        question = "Copy amounts exactly."
        expected = (f"Requested field schema:\n{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}\n\n"
                    f"Question: {question}\nKeep field names exactly as provided.")
        self.assertEqual(_build_prompt(question, SCHEMA, "baseline"), expected)

    def test_experiment_does_not_force_text_amounts_on_numeric_demo_schema(self):
        with self.assertRaisesRegex(ValueError, "string or null"):
            _build_prompt("Read total", {"total_amount": "number or null"}, "label-aware-v1")

    def test_invalid_profile_fails_instead_of_silently_using_baseline(self):
        with self.assertRaises(ValueError):
            _build_prompt("Read", SCHEMA, "label-aware-typo")

    def test_observed_invalid_outputs_are_flagged_without_reformatting(self):
        fields = {"tax_amount": "10%", "total_amount": "Rp 20,000", "cash_amount": 100000,
                  "change_amount": "80.000"}
        original = copy.deepcopy(fields)
        flags = amount_review_flags(fields, SCHEMA)
        self.assertEqual({flag["field"]: flag["reason"] for flag in flags},
                         {"tax_amount": "percentage_is_not_amount", "cash_amount": "amount_must_be_string"})
        self.assertEqual(fields, original)

    def test_null_zero_and_region_specific_separators_are_not_repaired(self):
        for text in (None, "0", "20,000", "20.000", "Rp 20.000", ":9,000", "0012"):
            self.assertEqual(amount_review_flags({"total_amount": text}, SCHEMA), [])

    def test_unicode_percentage_and_placeholders_are_flagged(self):
        for text in ("１０％", "null", "N/A", "", "unreadable"):
            self.assertEqual(len(amount_review_flags({"tax_amount": text}, SCHEMA)), 1)

    def test_numeric_nonreceipt_schema_is_not_subject_to_text_checks(self):
        self.assertEqual(amount_review_flags({"total_amount": 12}, {"total_amount": "number"}), [])

    def test_document_analyzer_retains_raw_value_and_adds_review_flags(self):
        analyzer = object.__new__(DocumentAnalyzer)
        analyzer.receipt_recheck = "none"
        analyzer._backend = Mock()
        analyzer._backend.analyze.return_value = {"fields": {"tax_amount": "10%"}, "raw_output": "original"}
        result = analyzer.analyze("image.png", "Read tax", SCHEMA)
        self.assertEqual(result["fields"]["tax_amount"], "10%")
        self.assertEqual(result["raw_output"], "original")
        self.assertEqual(len(result["review_flags"]), 1)
        self.assertEqual(analyzer._backend.analyze.call_args.args, ("image.png", "Read tax", SCHEMA))

    def test_echoed_format_template_is_a_parse_failure(self):
        with self.assertRaises(OutputParseError):
            parse_model_output('{"fields":{"requested_field":"printed amount or null"}}', list(SCHEMA))

    def test_recheck_only_changes_flagged_fields_and_retains_both_responses(self):
        first = {"fields": {"tax_amount": "10%", "total_amount": "20,000"}, "raw_output": "first raw",
                 "metadata": {"latency_seconds": 2.0, "peak_gpu_memory_gib": 2.4}}
        backend = Mock()
        backend.analyze.return_value = {"fields": {"tax_amount": "1,818"}, "raw_output": "second raw",
                                       "metadata": {"latency_seconds": 0.6, "peak_gpu_memory_gib": 2.5}}
        result = recheck_invalid_amounts(backend, "image.png", SCHEMA, first)
        self.assertEqual(result["fields"], {"tax_amount": "1,818", "total_amount": "20,000"})
        self.assertEqual(first["fields"]["tax_amount"], "10%")
        self.assertEqual(result["initial_prediction"]["raw_output"], "first raw")
        self.assertEqual(result["recheck_attempt"]["prediction"]["raw_output"], "second raw")
        self.assertEqual(backend.analyze.call_args.args[2], {"tax_amount": "string or null"})
        self.assertEqual(result["metadata"]["latency_seconds"], 2.6)
        self.assertEqual(result["metadata"]["peak_gpu_memory_gib"], 2.5)

    def test_failed_or_invalid_recheck_keeps_original_and_records_failure(self):
        first = {"fields": {"tax_amount": "10%"}, "raw_output": "original"}
        backend = Mock()
        backend.analyze.side_effect = RuntimeError("failed request")
        failed = recheck_invalid_amounts(backend, "image.png", SCHEMA, first)
        self.assertEqual(failed["fields"], first["fields"])
        self.assertIn("failed request", failed["recheck_attempt"]["error"])
        self.assertIsNone(failed["metadata"]["latency_seconds"])
        backend.analyze.side_effect = None
        backend.analyze.return_value = {"fields": {"tax_amount": "20%"}}
        invalid = recheck_invalid_amounts(backend, "image.png", SCHEMA, first)
        self.assertEqual(invalid["fields"], first["fields"])
        self.assertEqual(invalid["recheck_attempt"]["accepted_fields"], [])

    def test_valid_first_pass_does_not_trigger_extra_inference(self):
        backend = Mock()
        first = {"fields": {"tax_amount": "0", "total_amount": "Rp 20,000"}}
        result = recheck_invalid_amounts(backend, "image.png", SCHEMA, first)
        backend.analyze.assert_not_called()
        self.assertEqual(result["fields"], first["fields"])

    def test_crop_geometry_preserves_full_width_and_stays_inside_image(self):
        self.assertEqual(central_receipt_box(418, 985), (0, 98, 418, 887))
        self.assertEqual(central_receipt_box(1, 1), (0, 0, 1, 1))
        with self.assertRaises(ValueError):
            central_receipt_box(100, 0)

    def test_crop_policy_does_not_recheck_placeholder_or_numeric_amounts(self):
        backend = Mock()
        first = {"fields": {"tax_amount": "null", "total_amount": 20000}}
        result = recheck_invalid_amounts(backend, "image.png", SCHEMA, first, "percent-crop-v1")
        backend.analyze.assert_not_called()
        self.assertEqual(result["fields"], first["fields"])

    def test_context_recheck_requests_context_but_merges_only_flagged_fields(self):
        backend = Mock()
        first = {"fields": {"tax_amount": "10%", "total_amount": "20,000", "cash_amount": "100,000"}}
        backend.analyze.return_value = {"fields": {"tax_amount": "1,818", "total_amount": "100,000", "cash_amount": "80,000"}}
        result = recheck_invalid_amounts(backend, "image.png", SCHEMA, first, "label-context-v1", "original question")
        self.assertEqual(result["fields"], {"tax_amount": "1,818", "total_amount": "20,000", "cash_amount": "100,000"})
        self.assertEqual(backend.analyze.call_args.args, ("image.png", "original question", SCHEMA))
        self.assertEqual(backend.analyze.call_args.kwargs, {"receipt_profile": "label-aware-v1"})
        self.assertEqual(result["recheck_attempt"]["accepted_fields"], ["tax_amount"])

    def test_merged_result_does_not_reuse_stale_explanation_or_confidence(self):
        backend = Mock()
        first = {"fields": {"tax_amount": "10%", "total_amount": "20,000"},
                 "answer": "Tax is 10%", "confidence": 0.9,
                 "evidence": [{"field": "tax_amount", "text": "10%"},
                              {"field": "total_amount", "text": "Total 20,000"}]}
        backend.analyze.return_value = {"fields": {"tax_amount": "1,818"},
            "evidence": [{"field": "tax_amount", "text": "PB1 1,818"}]}
        result = recheck_invalid_amounts(backend, "image.png", SCHEMA, first)
        self.assertEqual(result["answer"], "")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["evidence"], [{"field": "total_amount", "text": "Total 20,000"},
                                             {"field": "tax_amount", "text": "PB1 1,818"}])
        self.assertEqual(result["initial_prediction"]["answer"], "Tax is 10%")

    def test_paired_report_exposes_improvements_and_regressions(self):
        baseline = {"dataset": {"selected_ids": ["receipt"]}, "results": [
            {"id": "receipt", "score": {"field_results": {"tax_amount": False, "total_amount": True}}}]}
        candidate = copy.deepcopy(baseline)
        candidate["results"][0]["score"]["field_results"] = {"tax_amount": True, "total_amount": False}
        self.assertEqual(paired_changes(baseline, candidate), {
            "tax_amount": {"improved": 1, "regressed": 0},
            "total_amount": {"improved": 0, "regressed": 1}})
        candidate["results"].append(candidate["results"][0])
        with self.assertRaises(ValueError):
            paired_changes(baseline, candidate)

    def test_offset_selects_disjoint_holdout_without_changing_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"fixture")
            dataset = root / "annotations.jsonl"
            dataset.write_text("".join(json.dumps({"id": str(index), "image": "image.png",
                "question": "Read amount", "field_schema": SCHEMA, "expected_fields": {"tax_amount": "1,818"},
                "mock_response": {"fields": {"tax_amount": "10%"},
                                  "review_flags": [{"field": "tax_amount", "reason": "percentage_is_not_amount", "value": "10%"}]}})
                + "\n" for index in range(4)), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--dataset", str(dataset), "--dry-run", "--limit", "2", "--output", str(root / "pilot")])
                main(["--dataset", str(dataset), "--dry-run", "--offset", "2", "--limit", "2", "--output", str(root / "holdout")])
            pilot = json.loads((root / "pilot/run.json").read_text())
            holdout = json.loads((root / "holdout/run.json").read_text())
            self.assertEqual(pilot["dataset"]["fingerprint"], holdout["dataset"]["fingerprint"])
            self.assertEqual(pilot["dataset"]["selected_ids"], ["0", "1"])
            self.assertEqual(holdout["dataset"]["selected_ids"], ["2", "3"])
            self.assertEqual(holdout["correct_fields"], 0)
            self.assertEqual(holdout["review_flagged_examples"], 2)
            self.assertIn("percentage_is_not_amount", (root / "holdout/review-flags.csv").read_text(encoding="utf-8-sig"))
            for offset in ("-1", "4"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--dataset", str(dataset), "--dry-run", "--offset", offset])


if __name__ == "__main__":
    unittest.main()
