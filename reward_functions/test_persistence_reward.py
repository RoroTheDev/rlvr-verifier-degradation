"""Regression tests for persistence_reward.py -- no GPU, no verl install needed.

Stubs out verl.utils.reward_score.default_compute_score with a controllable fake,
so this exercises only our own noise/persistence/targeting/coverage logic, not
verl's own verifiers.

Run: python reward_functions/test_persistence_reward.py
"""

import glob
import importlib
import json
import os
import random
import sys
import tempfile
import types
import unittest

ENV_KEYS = (
    "MIXUP_MODE",
    "MIXUP_TPR",
    "MIXUP_FPR",
    "MIXUP_COVERAGE",
    "MIXUP_MASK_SEED",
    "MIXUP_CORRECT_THRESHOLD",
    "MIXUP_TARGET_FIELD",
    "MIXUP_TARGET_OP",
    "MIXUP_TARGET_VALUE",
    "MIXUP_LOG_DIR",
)


def _install_fake_verl(honest_scores: dict):
    """honest_scores maps ground_truth -> the raw value verl's verifier would
    return: a float, a bool, or a dict with a "score" key. Anything not in the
    dict defaults to 1.0 (correct)."""

    def fake_default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        value = honest_scores.get(ground_truth, 1.0)
        return float(value) if isinstance(value, bool) else value

    fake_reward_score = types.ModuleType("verl.utils.reward_score")
    fake_reward_score.default_compute_score = fake_default_compute_score
    fake_utils = types.ModuleType("verl.utils")
    fake_utils.reward_score = fake_reward_score
    fake_verl = types.ModuleType("verl")
    fake_verl.utils = fake_utils

    sys.modules["verl"] = fake_verl
    sys.modules["verl.utils"] = fake_utils
    sys.modules["verl.utils.reward_score"] = fake_reward_score


def _load_module(env: dict, honest_scores: dict = None):
    """(Re)import persistence_reward under a given env config and fake verifier.
    Reloading also stands in for "a fresh Ray worker process importing the module"."""
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    _install_fake_verl(honest_scores or {})

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if "reward_functions.persistence_reward" in sys.modules:
        return importlib.reload(sys.modules["reward_functions.persistence_reward"])
    import reward_functions.persistence_reward as m

    return m


def _score(m, ground_truth, solution="s", extra_info=None):
    return m.compute_score("gsm8k", solution, ground_truth, extra_info=extra_info)["score"]


def _info(i, split="train"):
    return {"split": split, "index": i}


class TestCleanMode(unittest.TestCase):
    def test_clean_mode_returns_honest_label_and_zero_flags(self):
        m = _load_module({"MIXUP_MODE": "clean", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0"}, {"42": True, "43": False})
        right = m.compute_score("gsm8k", "s", "42")
        wrong = m.compute_score("gsm8k", "s", "43")
        self.assertEqual((right["score"], right["acc"]), (1.0, 1.0))
        self.assertEqual((wrong["score"], wrong["acc"]), (0.0, 0.0))
        for r in (right, wrong):
            self.assertEqual((r["mixup_eligible"], r["mixup_covered"], r["mixup_flipped"]), (0.0, 0.0, 0.0))

    def test_unknown_mode_raises(self):
        m = _load_module({"MIXUP_MODE": "bogus"})
        with self.assertRaises(ValueError):
            m.compute_score("gsm8k", "s", "42")


class TestPersistentDeterminism(unittest.TestCase):
    def test_same_task_same_result_every_time(self):
        """The W2 requirement: a mask must be identical epoch 1 to epoch 10."""
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5"},
            {"gt-A": True, "gt-B": False},
        )
        first_a = _score(m, "gt-A", "solution v1", _info(1))
        first_b = _score(m, "gt-B", "solution v1", _info(2))
        for epoch in range(10):
            # Different response text every epoch, as real rollouts produce.
            self.assertEqual(_score(m, "gt-A", f"rollout {epoch}", _info(1)), first_a)
            self.assertEqual(_score(m, "gt-B", f"rollout {epoch}", _info(2)), first_b)

    def test_determinism_survives_a_fresh_import(self):
        """Stands in for a different Ray worker process: no shared state, same answer."""
        env = {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5", "MIXUP_COVERAGE": "0.7"}
        m = _load_module(env, {})
        before = [_score(m, "x", "s", _info(i)) for i in range(100)]
        m = _load_module(env, {})  # fresh import, as another worker would do
        after = [_score(m, "x", "s", _info(i)) for i in range(100)]
        self.assertEqual(before, after)

    def test_different_tasks_get_different_results(self):
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.0"}, {})
        results = {_score(m, "x", "s", _info(i)) for i in range(50)}
        self.assertEqual(results, {0.0, 1.0})

    def test_same_answer_different_task_is_not_one_shared_coin(self):
        """Regression for the identity bug: GSM8K ground_truth is just the final
        number, so keying on it would flip every question answering "18" together."""
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.0"}, {})
        same_answer = {_score(m, "18", "s", _info(i)) for i in range(80)}
        self.assertEqual(same_answer, {0.0, 1.0}, "tasks sharing an answer must not share one flip")

    def test_split_is_part_of_identity(self):
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.0"}, {})
        train = [_score(m, "x", "s", _info(i, "train")) for i in range(100)]
        test = [_score(m, "x", "s", _info(i, "test")) for i in range(100)]
        self.assertNotEqual(train, test, "index 7 in train and index 7 in test are different tasks")


class TestMaskSeed(unittest.TestCase):
    def _mask(self, seed):
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0", "MIXUP_COVERAGE": "0.5", "MIXUP_MASK_SEED": str(seed)},
            {},
        )
        return [_score(m, "x", "s", _info(i)) for i in range(200)]

    def test_same_seed_same_mask(self):
        self.assertEqual(self._mask(3), self._mask(3))

    def test_different_seed_different_mask(self):
        self.assertNotEqual(self._mask(1), self._mask(2))


class TestResampledMode(unittest.TestCase):
    def test_resampled_varies_across_calls(self):
        m = _load_module({"MIXUP_MODE": "resampled", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5"}, {"gt-A": True})
        results = {_score(m, "gt-A", "s", _info(1)) for _ in range(200)}
        self.assertEqual(results, {0.0, 1.0})

    def test_resampled_ignores_global_rng_seed(self):
        """Workers often seed the global RNG identically; resampled noise must not
        collapse to the same sequence in every worker because of it."""
        m = _load_module({"MIXUP_MODE": "resampled", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5"}, {})
        random.seed(0)
        a = [_score(m, "x", "s", _info(1)) for _ in range(64)]
        random.seed(0)
        b = [_score(m, "x", "s", _info(1)) for _ in range(64)]
        self.assertNotEqual(a, b)


class TestTargeting(unittest.TestCase):
    ENV = {
        "MIXUP_MODE": "persistent",
        "MIXUP_TPR": "0.0",
        "MIXUP_FPR": "1.0",
        "MIXUP_TARGET_FIELD": "difficulty",
        "MIXUP_TARGET_OP": "eq",
        "MIXUP_TARGET_VALUE": "hard",
    }

    def test_untargeted_item_stays_honest(self):
        m = _load_module(self.ENV, {"gt-A": True})
        r = m.compute_score("gsm8k", "s", "gt-A", extra_info={"difficulty": "easy", "index": 1})
        self.assertEqual((r["score"], r["mixup_eligible"], r["mixup_flipped"]), (1.0, 0.0, 0.0))

    def test_targeted_item_gets_noise(self):
        m = _load_module(self.ENV, {"gt-A": True})
        r = m.compute_score("gsm8k", "s", "gt-A", extra_info={"difficulty": "hard", "index": 1})
        self.assertEqual((r["score"], r["mixup_eligible"], r["mixup_flipped"]), (0.0, 1.0, 1.0))

    def test_missing_field_is_not_eligible(self):
        m = _load_module(self.ENV, {"gt-A": True})
        r = m.compute_score("gsm8k", "s", "gt-A", extra_info={"index": 1})
        self.assertEqual(r["mixup_eligible"], 0.0)


class TestCoverage(unittest.TestCase):
    def test_coverage_fraction_and_stability(self):
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0", "MIXUP_COVERAGE": "0.5"}, {}
        )
        n = 400
        first = [_score(m, "x", "s", _info(i)) for i in range(n)]
        noisy = sum(1 for r in first if r == 0.0)
        self.assertTrue(0.35 * n < noisy < 0.65 * n, f"expected ~50% coverage, got {noisy}/{n}")
        second = [_score(m, "x", "different solution text", _info(i)) for i in range(n)]
        self.assertEqual(first, second)


class TestVerifierConventions(unittest.TestCase):
    """verl verifiers disagree on what a score looks like; correctness must not
    depend on which one a dataset happens to route to."""

    def test_dict_return_is_unwrapped(self):
        m = _load_module({"MIXUP_MODE": "clean"}, {"ok": {"score": 1.0, "acc": True, "pred": "5"}, "bad": {"score": 0.0}})
        self.assertEqual(m.compute_score("d", "s", "ok")["acc"], 1.0)
        self.assertEqual(m.compute_score("d", "s", "bad")["acc"], 0.0)

    def test_negative_one_scale_is_incorrect(self):
        """math_dapo returns -1.0 for wrong answers; bool(-1.0) is True, which would
        have mislabelled every wrong answer as correct."""
        m = _load_module({"MIXUP_MODE": "clean"}, {"ok": {"score": 1.0}, "bad": {"score": -1.0}})
        self.assertEqual(m.compute_score("d", "s", "ok")["score"], 1.0)
        self.assertEqual(m.compute_score("d", "s", "bad")["score"], 0.0)

    def test_partial_credit_is_incorrect_by_default(self):
        m = _load_module({"MIXUP_MODE": "clean"}, {"partial": 0.5, "full": 1.0})
        self.assertEqual(m.compute_score("d", "s", "partial")["acc"], 0.0)
        self.assertEqual(m.compute_score("d", "s", "full")["acc"], 1.0)

    def test_threshold_is_configurable(self):
        m = _load_module({"MIXUP_MODE": "clean", "MIXUP_CORRECT_THRESHOLD": "0.5"}, {"partial": 0.5})
        self.assertEqual(m.compute_score("d", "s", "partial")["acc"], 1.0)


class TestHonestAccuracyIsReported(unittest.TestCase):
    def test_acc_stays_honest_when_label_is_flipped(self):
        """verl stores a float return as "acc"; returning a dict lets acc stay the
        true correctness while score carries the corrupted label."""
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0"}, {"right": True, "wrong": False})
        r = m.compute_score("gsm8k", "s", "right", extra_info=_info(1))
        w = m.compute_score("gsm8k", "s", "wrong", extra_info=_info(2))
        self.assertEqual((r["acc"], r["score"], r["mixup_flipped"]), (1.0, 0.0, 1.0))
        self.assertEqual((w["acc"], w["score"], w["mixup_flipped"]), (0.0, 1.0, 1.0))


class TestCallLog(unittest.TestCase):
    def test_log_written_when_dir_set(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.9", "MIXUP_FPR": "0.2", "MIXUP_LOG_DIR": d}, {})
            for i in range(5):
                m.compute_score("gsm8k", "s", "x", extra_info=_info(i))
            for h in m._log_handles.values():
                h.close()
            files = glob.glob(os.path.join(d, "mixup_calls.*.jsonl"))
            self.assertEqual(len(files), 1)
            with open(files[0]) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["mode"] for r in rows}, {"persistent"})
            self.assertEqual({r["tpr"] for r in rows}, {0.9})
            for key in ("task", "honest_score", "score", "acc", "mixup_flipped", "mask_seed"):
                self.assertIn(key, rows[0])

    def test_no_log_without_dir(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.9", "MIXUP_FPR": "0.2"}, {})
            m.compute_score("gsm8k", "s", "x", extra_info=_info(1))
            self.assertEqual(glob.glob(os.path.join(d, "*")), [])
            self.assertEqual(m._log_handles, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
