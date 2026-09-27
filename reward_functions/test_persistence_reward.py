"""Regression test for persistence_reward.py -- no GPU, no verl install needed.

Stubs out verl.utils.reward_score.default_compute_score with a controllable
fake, so this exercises only our own noise/persistence/targeting/coverage
logic, not verl's own verifiers (not our code, not our job to test here).

Run: python reward_functions/test_persistence_reward.py
"""

import importlib
import os
import sys
import types
import unittest


def _install_fake_verl(honest_scores: dict):
    """honest_scores maps ground_truth -> True/False. Anything not in the
    dict defaults to True (correct), which is fine for tests that don't care."""

    def fake_default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        return 1.0 if honest_scores.get(ground_truth, True) else 0.0

    fake_reward_score = types.ModuleType("verl.utils.reward_score")
    fake_reward_score.default_compute_score = fake_default_compute_score
    fake_utils = types.ModuleType("verl.utils")
    fake_utils.reward_score = fake_reward_score
    fake_verl = types.ModuleType("verl")
    fake_verl.utils = fake_utils

    sys.modules["verl"] = fake_verl
    sys.modules["verl.utils"] = fake_utils
    sys.modules["verl.utils.reward_score"] = fake_reward_score


def _load_module(env: dict, honest_scores: dict):
    """(Re)import persistence_reward with a given env config and fake verifier."""
    for k in ("MIXUP_MODE", "MIXUP_TPR", "MIXUP_FPR", "MIXUP_COVERAGE", "MIXUP_TARGET_FIELD", "MIXUP_TARGET_OP", "MIXUP_TARGET_VALUE"):
        os.environ.pop(k, None)
    os.environ.update(env)

    _install_fake_verl(honest_scores)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "reward_functions.persistence_reward" in sys.modules:
        return importlib.reload(sys.modules["reward_functions.persistence_reward"])
    import reward_functions.persistence_reward as m

    return m


class TestCleanMode(unittest.TestCase):
    def test_clean_mode_returns_honest_score_unchanged(self):
        m = _load_module({"MIXUP_MODE": "clean"}, honest_scores={"42": True, "43": False})
        self.assertEqual(m.compute_score("gsm8k", "any solution", "42"), 1.0)
        self.assertEqual(m.compute_score("gsm8k", "any solution", "43"), 0.0)


class TestPersistentDeterminism(unittest.TestCase):
    def test_same_task_same_result_every_time(self):
        """The actual W2 requirement: a mask must be identical epoch 1 to epoch 10."""
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5", "MIXUP_COVERAGE": "1.0"},
            honest_scores={"task-A": True, "task-B": False},
        )
        first_a = m.compute_score("gsm8k", "solution v1", "task-A")
        first_b = m.compute_score("gsm8k", "solution v1", "task-B")
        # Simulate "epoch 10": call again with a totally different solution_str
        # (as a real rollout would produce), same task identity.
        for _ in range(10):
            self.assertEqual(m.compute_score("gsm8k", "a completely different solution", "task-A"), first_a)
            self.assertEqual(m.compute_score("gsm8k", "yet another one", "task-B"), first_b)

    def test_different_tasks_can_get_different_results(self):
        """Sanity check: determinism shouldn't mean everything maps to the same coin."""
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "1.0", "MIXUP_FPR": "1.0", "MIXUP_COVERAGE": "1.0"},
            honest_scores={},
        )
        # honest is always True here (default), FPR is irrelevant; TPR=1.0 means
        # everything stays correct regardless of task -- use TPR<1 instead so we
        # can see real variation across many distinct tasks.
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.0", "MIXUP_COVERAGE": "1.0"},
            honest_scores={},
        )
        results = {m.compute_score("gsm8k", "s", f"task-{i}") for i in range(50)}
        self.assertTrue({0.0, 1.0}.issubset(results), "expected both flipped and unflipped outcomes across 50 distinct tasks")


class TestResampledMode(unittest.TestCase):
    def test_resampled_varies_across_calls(self):
        m = _load_module(
            {"MIXUP_MODE": "resampled", "MIXUP_TPR": "0.5", "MIXUP_FPR": "0.5", "MIXUP_COVERAGE": "1.0"},
            honest_scores={"task-A": True},
        )
        results = {m.compute_score("gsm8k", "s", "task-A") for _ in range(200)}
        # With p=0.5 over 200 draws, P(all identical) ~ 2 * 0.5^200 -- not happening by chance.
        self.assertEqual(results, {0.0, 1.0}, "expected resampled mode to actually vary across calls")


class TestTargeting(unittest.TestCase):
    def test_untargeted_item_stays_honest(self):
        m = _load_module(
            {
                "MIXUP_MODE": "persistent",
                "MIXUP_TPR": "0.0",
                "MIXUP_FPR": "1.0",  # would flip everything if eligible
                "MIXUP_COVERAGE": "1.0",
                "MIXUP_TARGET_FIELD": "difficulty",
                "MIXUP_TARGET_OP": "eq",
                "MIXUP_TARGET_VALUE": "hard",
            },
            honest_scores={"task-A": True},
        )
        # extra_info doesn't match the selector -> must stay honest despite TPR=0.
        result = m.compute_score("gsm8k", "s", "task-A", extra_info={"difficulty": "easy"})
        self.assertEqual(result, 1.0)

    def test_targeted_item_gets_noise(self):
        m = _load_module(
            {
                "MIXUP_MODE": "persistent",
                "MIXUP_TPR": "0.0",
                "MIXUP_FPR": "1.0",
                "MIXUP_COVERAGE": "1.0",
                "MIXUP_TARGET_FIELD": "difficulty",
                "MIXUP_TARGET_OP": "eq",
                "MIXUP_TARGET_VALUE": "hard",
            },
            honest_scores={"task-A": True},
        )
        # matches the selector, TPR=0.0 -> a correct answer gets flipped to wrong.
        result = m.compute_score("gsm8k", "s", "task-A", extra_info={"difficulty": "hard"})
        self.assertEqual(result, 0.0)


class TestCoverage(unittest.TestCase):
    def test_coverage_fraction_and_stability(self):
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0", "MIXUP_COVERAGE": "0.5"},
            honest_scores={},  # honest = True (correct) for everything
        )
        n = 400
        first_pass = [m.compute_score("gsm8k", "s", f"task-{i}") for i in range(n)]
        noisy_count = sum(1 for r in first_pass if r == 0.0)
        # Roughly half should be covered (flipped); loose bounds to avoid flakiness.
        self.assertTrue(0.35 * n < noisy_count < 0.65 * n, f"expected ~50% coverage, got {noisy_count}/{n}")

        # Stability: re-scoring the same tasks reproduces exactly the same covered set.
        second_pass = [m.compute_score("gsm8k", "different solution text", f"task-{i}") for i in range(n)]
        self.assertEqual(first_pass, second_pass)


if __name__ == "__main__":
    unittest.main(verbosity=2)
