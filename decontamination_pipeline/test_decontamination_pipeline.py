from __future__ import annotations

from json import loads
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from pathlib import Path

from decontamination_pipeline.decontamination_pipeline import run


class DecontaminationPipelineTests(TestCase):
    def test_removes_exact_and_fuzzy_test_overlaps_from_training_output(self) -> None:
        words = [f"word{i}" for i in range(80)]
        reference = " ".join(words)
        near_duplicate_words = words.copy()
        near_duplicate_words[40] = "different"
        near_duplicate = " ".join(near_duplicate_words)
        training_rows = [
            {"question": reference.upper(), "answer": "exact"},
            {"question": near_duplicate, "answer": "fuzzy"},
            {"question": "an unrelated training problem", "answer": "keep"},
        ]

        test_rows = [
            {"problem": reference, "solution": "original solution", "answer": "42",
             "subject": "Algebra", "level": 1, "unique_id": "test/1"},
            {"problem": "a separate evaluation question", "answer": "7"},
        ]
        with TemporaryDirectory() as output_dir:
            run(
                math500=test_rows,
                gsm8k=training_rows,
                output_dir=output_dir,
            )

            output_path = Path(output_dir) / "clean_gsm8k_train.jsonl"
            report_path = Path(output_dir) / "report.json"
            clean_rows = [
                loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            report = loads(report_path.read_text(encoding="utf-8"))
            exported_test = [
                loads(line)
                for line in (Path(output_dir) / "math500_test.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            markdown = (Path(output_dir) / "decontamination_report.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(exported_test, test_rows)
        self.assertEqual(report["total_math500"], 2)
        self.assertEqual(report["retained_math500"], 2)
        self.assertIn("2/2 examples retained", markdown)
        self.assertIn("3 − 2 =\n  **1 examples**", markdown)
        self.assertIn("0.80", markdown)
        self.assertIn("Prompt formatting", markdown)
        self.assertEqual(clean_rows, [training_rows[2]])
        self.assertEqual(report["removed_exact"], 1)
        self.assertEqual(report["removed_fuzzy"], 1)
        self.assertEqual(report["removed_total"], 2)
        self.assertEqual(report["clean"], 1)

    def test_zero_removal_exports_original_records(self) -> None:
        training_rows = [{"question": "unrelated training question", "answer": "é"}]
        test_rows = [{"problem": "distinct test problem", "solution": "unchanged"}]
        with TemporaryDirectory() as output_dir:
            run(math500=test_rows, gsm8k=training_rows, output_dir=output_dir)
            output = Path(output_dir)
            self.assertEqual(
                [loads(line) for line in (output / "clean_gsm8k_train.jsonl")
                 .read_text(encoding="utf-8").splitlines()],
                training_rows,
            )
            report = loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["removed_total"], 0)
            self.assertEqual(report["clean"], 1)
            markdown = (output / "decontamination_report.md").read_text(encoding="utf-8")
            self.assertIn("1 − 0 =\n  **1 examples**", markdown)
            self.assertIn("semantic judge was not used", markdown)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            run(threshold=1.1, math500=[], gsm8k=[])


if __name__ == "__main__":
    main()
