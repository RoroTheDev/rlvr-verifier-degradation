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
    "MIXUP_GROUP_P",
    "MIXUP_REWARD_SCALE",
    "MIXUP_CLEAN_SPLITS",
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
        # "test" is a clean split now, so use a different *noisy* split name here.
        other = [_score(m, "x", "s", _info(i, "train_b")) for i in range(100)]
        self.assertNotEqual(train, other, "index 7 in train and index 7 in train_b are different tasks")


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


class TestTrainOnlyNoise(unittest.TestCase):
    """Validation must be measured against the real verifier even when training is corrupted."""

    def test_clean_splits_are_never_corrupted_in_any_mode(self):
        honest = {"gt-A": True, "gt-B": False}
        modes = [
            {"MIXUP_MODE": "resampled", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0"},
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0"},
            {"MIXUP_MODE": "group_resampled", "MIXUP_GROUP_P": "1.0"},
            {"MIXUP_MODE": "group_persistent", "MIXUP_GROUP_P": "1.0"},
        ]
        for env in modes:
            m = _load_module(env, honest)
            for split in ("validation", "val", "test"):
                info = {"split": split, "index": 1, "encounter": 0}
                a = m.compute_score("gsm8k", "s", "gt-A", extra_info=info)
                b = m.compute_score("gsm8k", "s", "gt-B", extra_info=info)
                self.assertEqual((a["score"], b["score"]), (1.0, 0.0), (env, split))
                for r in (a, b):
                    self.assertEqual((r["mixup_eligible"], r["mixup_covered"], r["mixup_flipped"]), (0.0, 0.0, 0.0))
            train = m.compute_score("gsm8k", "s", "gt-A", extra_info={"split": "train", "index": 1, "encounter": 0})
            self.assertEqual(train["score"], 0.0, f"{env}: the same row is corrupted when it is training data")

    def test_missing_split_is_treated_as_training_data(self):
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0"}, {"gt-A": True})
        self.assertEqual(m.compute_score("gsm8k", "s", "gt-A", extra_info={"index": 1})["score"], 0.0)

    def test_clean_splits_are_configurable(self):
        m = _load_module(
            {"MIXUP_MODE": "persistent", "MIXUP_TPR": "0.0", "MIXUP_FPR": "1.0", "MIXUP_CLEAN_SPLITS": "holdout"},
            {"gt-A": True},
        )
        self.assertEqual(m.compute_score("gsm8k", "s", "gt-A", extra_info=_info(1, "holdout"))["score"], 1.0)
        self.assertEqual(m.compute_score("gsm8k", "s", "gt-A", extra_info=_info(1, "test"))["score"], 0.0)


def _g(m, task, encounter, gt="gt", split="train", solution="s", **extra):
    info = {"split": split, "index": task, "encounter": encounter}
    info.update(extra)
    return m.compute_score("gsm8k", solution, gt, extra_info=info)


def _group_env(mode, p, **more):
    env = {"MIXUP_MODE": mode, "MIXUP_GROUP_P": str(p)}
    env.update(more)
    return env


class TestGroupNoise(unittest.TestCase):
    def test_p1_inverts_the_reward_and_p0_leaves_it(self):
        honest = {"hi": 1.0, "lo": 0.0, "mid": 0.75}
        for mode in ("group_resampled", "group_persistent"):
            m = _load_module(_group_env(mode, 1.0), honest)
            self.assertEqual([_g(m, 1, 0, gt)["score"] for gt in ("hi", "lo", "mid")], [0.0, 1.0, 0.25])
            m = _load_module(_group_env(mode, 0.0), honest)
            r = _g(m, 1, 0, "mid")
            self.assertEqual((r["score"], r["mixup_flipped"]), (0.75, 0.0))

    def test_format_failure_is_never_inverted(self):
        for mode in ("group_resampled", "group_persistent"):
            m = _load_module(_group_env(mode, 1.0), {"nocode": -0.25})
            r = _g(m, 1, 0, "nocode")
            self.assertEqual((r["score"], r["mixup_flipped"]), (-0.25, 0.0))

    def test_acc_stays_honest_when_the_group_is_inverted(self):
        m = _load_module(_group_env("group_persistent", 1.0), {"hi": 1.0, "lo": 0.0})
        hi, lo = _g(m, 1, 0, "hi"), _g(m, 1, 0, "lo")
        self.assertEqual((hi["score"], hi["acc"], hi["honest_reward"]), (0.0, 1.0, 1.0))
        self.assertEqual((lo["score"], lo["acc"], lo["honest_reward"]), (1.0, 0.0, 0.0))

    def test_every_rollout_of_a_group_shares_one_coin(self):
        """Different rollouts of one prompt have different honest rewards but must be
        inverted together or not at all -- that is what 'whole matrix' means."""
        honest = {"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.25}
        for mode in ("group_resampled", "group_persistent"):
            m = _load_module(_group_env(mode, 0.5), honest)
            seen = set()
            for task in range(300):
                # Real rollouts of a prompt have different text; the coin must not depend on it.
                flags = {_g(m, task, 3, gt, solution=f"rollout text for {gt}")["mixup_flipped"] for gt in honest}
                self.assertEqual(len(flags), 1, f"{mode}: task {task} was inverted for only some rollouts")
                seen |= flags
            self.assertEqual(seen, {0.0, 1.0})

    def test_marginal_rate_matches_p_in_both_modes(self):
        for mode in ("group_resampled", "group_persistent"):
            m = _load_module(_group_env(mode, 0.15), {})
            rate = sum(_g(m, t, 0)["mixup_flipped"] for t in range(4000)) / 4000
            self.assertTrue(0.12 < rate < 0.18, f"{mode}: flip rate {rate}")

    def test_group_resampled_is_deterministic_per_task_and_encounter(self):
        env = _group_env("group_resampled", 0.5)
        m = _load_module(env, {})
        first = [_g(m, t, 2)["mixup_flipped"] for t in range(200)]
        m = _load_module(env, {})  # fresh import = another Ray worker
        self.assertEqual(first, [_g(m, t, 2)["mixup_flipped"] for t in range(200)])

    def test_group_resampled_redraws_on_every_encounter(self):
        m = _load_module(_group_env("group_resampled", 0.5), {})
        e0 = [_g(m, t, 0)["mixup_flipped"] for t in range(400)]
        e1 = [_g(m, t, 1)["mixup_flipped"] for t in range(400)]
        differ = sum(a != b for a, b in zip(e0, e1)) / 400
        self.assertTrue(0.35 < differ < 0.65, f"encounters look correlated: {differ}")
        per_task = [{_g(m, t, k)["mixup_flipped"] for k in range(10)} for t in range(50)]
        self.assertGreater(sum(len(s) == 2 for s in per_task), 40, "a task should flip on some encounters, not others")

    def test_group_persistent_is_identical_on_every_encounter(self):
        """The persistence requirement, for the group form: epoch 1 == epoch 10."""
        env = _group_env("group_persistent", 0.5)
        m = _load_module(env, {})
        first = [_g(m, t, 0)["mixup_flipped"] for t in range(200)]
        for encounter in range(1, 10):
            self.assertEqual(first, [_g(m, t, encounter)["mixup_flipped"] for t in range(200)])
        m = _load_module(env, {})
        self.assertEqual(first, [_g(m, t, 0)["mixup_flipped"] for t in range(200)])

    def test_group_persistent_needs_no_encounter_but_resampled_refuses_without_one(self):
        m = _load_module(_group_env("group_persistent", 0.5), {})
        m.compute_score("gsm8k", "s", "gt", extra_info={"split": "train", "index": 1})
        m = _load_module(_group_env("group_resampled", 0.5), {})
        with self.assertRaises(ValueError):
            m.compute_score("gsm8k", "s", "gt", extra_info={"split": "train", "index": 1})

    def test_mask_seed_changes_the_persistent_mask(self):
        def mask(seed):
            m = _load_module(_group_env("group_persistent", 0.5, MIXUP_MASK_SEED=str(seed)), {})
            return [_g(m, t, 0)["mixup_flipped"] for t in range(200)]

        self.assertEqual(mask(3), mask(3))
        self.assertNotEqual(mask(3), mask(4))

    def test_targeting_still_gates_group_noise(self):
        env = _group_env("group_persistent", 1.0, MIXUP_TARGET_FIELD="difficulty", MIXUP_TARGET_VALUE="hard")
        m = _load_module(env, {"hi": 1.0})
        self.assertEqual(_g(m, 1, 0, "hi", difficulty="easy")["score"], 1.0)
        self.assertEqual(_g(m, 1, 0, "hi", difficulty="hard")["score"], 0.0)


class TestRewardScale(unittest.TestCase):
    def test_clean_continuous_keeps_partial_credit_and_binary_stays_the_default(self):
        m = _load_module({"MIXUP_MODE": "clean", "MIXUP_REWARD_SCALE": "continuous"}, {"half": 0.5})
        self.assertEqual(m.compute_score("gsm8k", "s", "half")["score"], 0.5)
        m = _load_module({"MIXUP_MODE": "clean"}, {"half": 0.5})
        self.assertEqual(m.compute_score("gsm8k", "s", "half")["score"], 0.0)

    def test_continuous_is_refused_for_the_binary_label_modes(self):
        m = _load_module({"MIXUP_MODE": "persistent", "MIXUP_REWARD_SCALE": "continuous"}, {})
        with self.assertRaises(ValueError):
            m.compute_score("gsm8k", "s", "x", extra_info=_info(1))

    def test_unknown_scale_raises(self):
        m = _load_module({"MIXUP_MODE": "clean", "MIXUP_REWARD_SCALE": "bogus"}, {})
        with self.assertRaises(ValueError):
            m.compute_score("gsm8k", "s", "x")


class TestMbppDispatch(unittest.TestCase):
    GT = json.dumps({"tests": ["assert add(1, 2) == 3", "assert add(2, 2) == 4"], "setup": ""})

    def test_mbpp_is_scored_by_our_scorer_not_verls_default(self):
        # The fake default scorer calls everything correct (1.0); only our own
        # scorer can return 0.0 for a wrong program.
        m = _load_module({"MIXUP_MODE": "clean", "MIXUP_REWARD_SCALE": "continuous"}, {})
        wrong = m.compute_score("mbpp", "```python\ndef add(a, b):\n    return 0\n```", self.GT, extra_info=_info(1))
        right = m.compute_score("mbpp", "```python\ndef add(a, b):\n    return a + b\n```", self.GT, extra_info=_info(1))
        self.assertEqual((wrong["honest_reward"], wrong["acc"]), (0.0, 0.0))
        self.assertEqual((right["honest_reward"], right["acc"]), (1.0, 1.0))

    def test_group_flip_on_a_real_mbpp_response(self):
        m = _load_module(_group_env("group_persistent", 1.0), {})
        r = m.compute_score("mbpp", "```python\ndef add(a, b):\n    return a + b\n```", self.GT, extra_info=_info(1))
        self.assertEqual((r["score"], r["acc"], r["mixup_flipped"]), (0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
