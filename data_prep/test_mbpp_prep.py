"""Tests for mbpp_prep.py's pure functions (no network, no datasets package).

Run: python data_prep/test_mbpp_prep.py
"""

import collections
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbpp_prep as p  # noqa: E402


def item(task_id, text="Write a function to add two numbers.", tests=None, setup=""):
    return {
        "task_id": task_id,
        "text": text,
        "test_list": tests or ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 0) == 0"],
        "test_setup_code": setup,
    }


class TestPrompt(unittest.TestCase):
    def test_function_name_is_found_in_the_usual_assert_shapes(self):
        cases = {
            "assert add(1, 2) == 3": "add",
            "assert remove_Occ('hello','l') == 'heo'": "remove_Occ",
            "assert not is_odd(2)": "is_odd",
            "assert set(get_pairs([1, 2])) == {(1, 2)}": "get_pairs",
            "assert math.isclose(area(2), 12.5)": "area",
        }
        for test, name in cases.items():
            self.assertEqual(p.parse_function_name(test), name, test)

    def test_prompt_names_the_function_shows_one_example_and_asks_for_a_fence(self):
        prompt = p.build_prompt("Add two numbers.", item(1)["test_list"])
        self.assertIn("Add two numbers.", prompt)
        self.assertIn("`add`", prompt)
        self.assertIn("assert add(1, 2) == 3", prompt)
        self.assertNotIn("assert add(2, 2) == 4", prompt, "only the first test is shown as the example")
        self.assertIn("```python", prompt)

    def test_prompt_still_builds_when_no_name_can_be_parsed(self):
        prompt = p.build_prompt("Do a thing.", ["assert 1 == 1"])
        self.assertNotIn("must be named", prompt)
        self.assertIn("```python", prompt)


class TestRows(unittest.TestCase):
    def test_row_has_the_layout_verl_expects(self):
        row = p.make_row(item(17, setup="import math"), "train", 4)
        self.assertEqual(row["data_source"], "mbpp")
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["reward_model"]["style"], "rule")
        self.assertEqual(row["extra_info"], {"split": "train", "index": 17, "encounter": 4})
        truth = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(truth["setup"], "import math")
        self.assertEqual(len(truth["tests"]), 3)

    def test_validation_rows_are_split_validation_encounter_zero(self):
        _, val = p.build_rows([item(1)], [item(10), item(11)], epochs=2, order_seed=0)
        self.assertEqual([r["extra_info"] for r in val], [
            {"split": "validation", "index": 10, "encounter": 0},
            {"split": "validation", "index": 11, "encounter": 0},
        ])


class TestBuckets(unittest.TestCase):
    def _csv(self, text):
        import tempfile

        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_csv_columns_become_typed_extra_info_fields(self):
        b = p.load_buckets(self._csv("task_id,bucket,difficulty,leaky\n11,hard,0.75,1\n12,easy,0.2,0\n"))
        self.assertEqual(b[11], {"bucket": "hard", "difficulty": 0.75, "leaky": 1})
        self.assertEqual(b[12]["bucket"], "easy")

    def test_rows_carry_the_bucket_but_it_cannot_overwrite_core_fields(self):
        row = p.make_row(item(5), "train", 2, {"bucket": "hard", "split": "evil", "index": 999, "encounter": 77})
        self.assertEqual(row["extra_info"], {"split": "train", "index": 5, "encounter": 2, "bucket": "hard"})

    def test_build_rows_applies_buckets_to_train_and_validation_and_skips_unlabelled(self):
        train, val = p.build_rows([item(1), item(2)], [item(10)], epochs=2, order_seed=0, buckets={1: {"bucket": "hard"}, 10: {"bucket": "easy"}})
        t = {(r["extra_info"]["index"], r["extra_info"]["encounter"]): r["extra_info"].get("bucket") for r in train}
        self.assertEqual({k: v for k, v in t.items() if k[0] == 1}, {(1, 0): "hard", (1, 1): "hard"})
        self.assertEqual({k: v for k, v in t.items() if k[0] == 2}, {(2, 0): None, (2, 1): None})
        self.assertEqual(val[0]["extra_info"]["bucket"], "easy")


class TestEpochOrder(unittest.TestCase):
    def test_every_pass_is_a_permutation_with_its_own_encounter_number(self):
        ids = list(range(1, 375))
        order = p.epoch_order(ids, epochs=5, order_seed=3)
        self.assertEqual(len(order), 5 * 374)
        for k in range(5):
            block = order[k * 374:(k + 1) * 374]
            self.assertEqual({e for _, e in block}, {k})
            self.assertEqual(sorted(t for t, _ in block), ids)

    def test_each_task_is_seen_once_per_encounter(self):
        order = p.epoch_order(range(50), epochs=7, order_seed=1)
        seen = collections.Counter(order)
        self.assertEqual(set(seen.values()), {1})
        self.assertEqual(len(seen), 50 * 7)

    def test_same_seed_same_order_different_seed_different_order(self):
        a = p.epoch_order(range(100), 3, order_seed=5)
        self.assertEqual(a, p.epoch_order(range(100), 3, order_seed=5))
        self.assertNotEqual(a, p.epoch_order(range(100), 3, order_seed=6))

    def test_passes_are_not_all_the_same_order(self):
        order = p.epoch_order(range(100), 2, order_seed=0)
        self.assertNotEqual([t for t, _ in order[:100]], [t for t, _ in order[100:]])

    def test_build_rows_train_matches_the_order(self):
        train, _ = p.build_rows([item(i) for i in range(1, 11)], [item(99)], epochs=3, order_seed=2)
        self.assertEqual(len(train), 30)
        self.assertEqual(
            [(r["extra_info"]["index"], r["extra_info"]["encounter"]) for r in train],
            p.epoch_order(range(1, 11), 3, 2),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
