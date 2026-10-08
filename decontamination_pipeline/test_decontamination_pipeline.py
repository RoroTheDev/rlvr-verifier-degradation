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

        with TemporaryDirectory() as output_dir:
            run(
                math500=[{"problem": reference}],
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

        self.assertEqual(clean_rows, [training_rows[2]])
        self.assertEqual(report["removed_exact"], 1)
        self.assertEqual(report["removed_fuzzy"], 1)
        self.assertEqual(report["removed_total"], 2)
        self.assertEqual(report["clean"], 1)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            run(threshold=1.1, math500=[], gsm8k=[])


if __name__ == "__main__":
    main()
