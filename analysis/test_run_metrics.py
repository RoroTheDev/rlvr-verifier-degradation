"""Tests for run_metrics.py on synthetic logs with hand-computable answers.

Run: python analysis/test_run_metrics.py
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics as m  # noqa: E402


def rec(task, enc, honest, score=None, flipped=0.0, ts=0.0, split="train", index=None):
    score = honest if score is None else score
    return {
        "task": task, "encounter": enc, "ts": ts, "split": split,
        "index": index if index is not None else (int(task[1:]) if task[1:].isdigit() else 0),
        "honest_reward": honest, "honest_score": honest, "score": score,
        "acc": 1.0 if honest >= 1.0 else 0.0, "mixup_flipped": flipped,
    }


def group(task, enc, honests, ts, scores=None, flipped=0.0, **kw):
    scores = scores or honests
    return [rec(task, enc, h, s, flipped, ts + i * 0.001, **kw) for i, (h, s) in enumerate(zip(honests, scores))]


class TestPassAtK(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(m.pass_at_k(16, 0, 1), 0.0)
        self.assertEqual(m.pass_at_k(16, 16, 8), 1.0)
        self.assertAlmostEqual(m.pass_at_k(4, 1, 1), 0.25)
        self.assertAlmostEqual(m.pass_at_k(4, 1, 2), 1 - math.comb(3, 2) / math.comb(4, 2))  # 0.5
        self.assertEqual(m.pass_at_k(4, 3, 2), 1.0)  # fewer than k wrong ones left: always at least one right


class TestSummary(unittest.TestCase):
    def test_hand_computed_group_metrics(self):
        g1 = group("t1", 0, [1.0, 1.0, 0.0, 0.0], ts=1)   # honest var 0.25, mixed
        g2 = group("t2", 0, [0.0, 0.0, 0.0, 0.0], ts=2)   # all fail: degenerate honestly
        s = m.summarise([("t1", 0, g1), ("t2", 0, g2)])
        self.assertEqual(s["n_groups"], 2)
        self.assertAlmostEqual(s["honest_pass1"], 2 / 8)
        self.assertAlmostEqual(s["reward_var_honest"], (0.25 + 0.0) / 2)
        self.assertAlmostEqual(s["degenerate_honest"], 0.5)
        self.assertAlmostEqual(s["adv_denominator"], (m._std([1, 1, 0, 0]) + 0.0) / 2)
        self.assertAlmostEqual(s["pass_at_1"], (0.5 + 0.0) / 2)
        self.assertAlmostEqual(s["pass_at_4"], (1.0 + 0.0) / 2)

    def test_inversion_can_remove_signal_or_add_it(self):
        # An inverted all-fail group becomes all-pass: still degenerate for the trainer. A
        # mixed group stays mixed when inverted (1-x), so the trainer's std is unchanged.
        mixed = group("a", 0, [1.0, 0.0, 1.0, 0.0], ts=1, scores=[0.0, 1.0, 0.0, 1.0], flipped=1.0)
        allfail = group("b", 0, [0.0] * 4, ts=2, scores=[1.0] * 4, flipped=1.0)
        s = m.summarise([("a", 0, mixed), ("b", 0, allfail)])
        self.assertAlmostEqual(s["inverted_frac"], 1.0)
        self.assertAlmostEqual(s["degenerate_noisy"], 0.5)
        self.assertAlmostEqual(s["honest_pass1"], 2 / 8)
        self.assertAlmostEqual(s["noisy_reward"], (2 + 4) / 8)

    def test_the_advantage_denominator_is_the_trainers_reward_not_the_honest_one(self):
        """Per-rollout noise can create variance in a group whose honest rewards are all equal:
        the trainer then gets a gradient from pure noise, which is exactly what this exposes."""
        g = group("a", 0, [0.0, 0.0, 0.0, 0.0], ts=1, scores=[1.0, 0.0, 1.0, 0.0], flipped=1.0)
        s = m.summarise([("a", 0, g)])
        self.assertAlmostEqual(s["adv_denominator"], m._std([1, 0, 1, 0]))
        self.assertGreater(s["adv_denominator"], 0.5)
        self.assertEqual((s["degenerate_noisy"], s["degenerate_honest"]), (0.0, 1.0))
        self.assertEqual((s["reward_var_honest"], s["reward_var_noisy"] > 0.2), (0.0, True))

    def test_format_failures_count_in_the_reward_but_not_as_correct(self):
        s = m.summarise([("a", 0, group("a", 0, [-0.25, 1.0], ts=1))])
        self.assertAlmostEqual(s["honest_reward"], 0.375)
        self.assertAlmostEqual(s["honest_pass1"], 0.5)


class TestStepsAndRounds(unittest.TestCase):
    def test_steps_are_recovered_from_scoring_order(self):
        recs = []
        for i in range(6):  # 6 groups; steps of 3 groups each, scored in two bursts
            recs += group(f"t{i}", 0, [1.0, 0.0], ts=(100 if i < 3 else 500) + i)
        rows = m.analyse(recs, batch_size=3)
        steps = [r for r in rows if r["kind"] == "train"]
        self.assertEqual([r["step"] for r in steps], [1, 2])
        self.assertEqual([r["n_groups"] for r in steps], [3, 3])

    def test_validation_is_separated_and_split_into_rounds(self):
        recs = group("t1", 0, [1.0, 0.0], ts=10)
        recs += group("v1", 0, [1.0, 1.0], ts=1000, split="validation") + group("v2", 0, [0.0, 0.0], ts=1001, split="validation")
        recs += group("v1", 0, [1.0, 0.0], ts=5000, split="validation")
        rows = m.analyse(recs, batch_size=48, round_gap=120)
        val = [r for r in rows if r["kind"] == "validation"]
        self.assertEqual([r["n_groups"] for r in val], [2, 1])
        self.assertAlmostEqual(val[0]["honest_pass1"], 0.5)
        train = [r for r in rows if r["kind"] == "train"]
        self.assertEqual([(r["n_groups"], r["n_calls"]) for r in train], [(1, 2)], "validation rows must not count as training")
        self.assertAlmostEqual(train[0]["honest_pass1"], 0.5)

    def test_same_task_on_different_encounters_is_two_groups(self):
        recs = group("t1", 0, [1.0, 0.0], ts=1) + group("t1", 1, [0.0, 0.0], ts=2)
        self.assertEqual(m.summarise(m.group_records(recs))["n_groups"], 2)


class TestBuckets(unittest.TestCase):
    def test_per_bucket_rows(self):
        recs = group("t1", 0, [1.0, 1.0], ts=1, index=1) + group("t2", 0, [0.0, 0.0], ts=2, index=2) + group("t3", 0, [0.0, 1.0], ts=3, index=3)
        rows = m.analyse(recs, batch_size=48, buckets={1: "easy", 2: "hard", 3: "hard"})
        by = {r["bucket"]: r for r in rows if r["kind"] == "train" and r["bucket"]}
        self.assertEqual(set(by), {"easy", "hard"})
        self.assertAlmostEqual(by["easy"]["honest_pass1"], 1.0)
        self.assertAlmostEqual(by["hard"]["honest_pass1"], 0.25)
        self.assertEqual(by["hard"]["n_groups"], 2)

    def test_unlabelled_tasks_do_not_crash(self):
        rows = m.analyse(group("t9", 0, [1.0, 0.0], ts=1, index=9), batch_size=48, buckets={1: "easy"})
        self.assertIn("(none)", {r["bucket"] for r in rows})


if __name__ == "__main__":
    unittest.main(verbosity=2)
