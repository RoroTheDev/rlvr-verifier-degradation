"""Tests for w1_gate.py: the statistics against known values, the decision rule, and the whole
script on small synthetic run directories.

Run: python analysis/test_w1_gate.py
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_gate as g  # noqa: E402


class TestStatistics(unittest.TestCase):
    def test_t_quantiles_match_published_tables(self):
        for df, expected in ((1, 6.3138), (2, 2.9200), (3, 2.3534), (4, 2.1318), (10, 1.8125), (30, 1.6973)):
            self.assertAlmostEqual(g.t_quantile(0.95, df), expected, places=3)
        self.assertAlmostEqual(g.t_quantile(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(g.t_cdf(0.0, 3.7), 0.5, places=12)
        self.assertTrue(g.t_quantile(0.95, 2.0) > g.t_quantile(0.95, 2.5) > g.t_quantile(0.95, 3.0))  # fractional df, as Welch gives

    def test_welch_by_hand(self):
        # clean 60, 62, 64: mean 62, var 4.  noisy 59, 61, 66: mean 62, var 13.
        w = g.welch([60.0, 62.0, 64.0], [59.0, 61.0, 66.0])
        self.assertAlmostEqual(w["D"], 0.0)
        self.assertAlmostEqual(w["s_clean"], 2.0)
        self.assertAlmostEqual(w["s_noisy"], math.sqrt(13.0))
        self.assertAlmostEqual(w["s_p"], math.sqrt((4.0 + 13.0) / 2.0))
        se2 = 4.0 / 3 + 13.0 / 3
        self.assertAlmostEqual(w["se"], math.sqrt(se2))
        self.assertAlmostEqual(w["df"], se2 ** 2 / ((4.0 / 3) ** 2 / 2 + (13.0 / 3) ** 2 / 2))
        self.assertAlmostEqual(w["UB"], g.t_quantile(0.95, w["df"]) * math.sqrt(se2))

    def test_equal_variances_give_df_four(self):
        w = g.welch([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertAlmostEqual(w["df"], 4.0)
        self.assertAlmostEqual(w["D"], -1.0)
        self.assertAlmostEqual(w["UB"], -1.0 + 2.1318 * math.sqrt(2.0 / 3.0), places=3)

    def test_identical_seeds_do_not_crash(self):
        w = g.welch([5.0, 5.0, 5.0], [4.0, 4.0, 4.0])
        self.assertEqual((w["D"], w["UB"], w["s_p"]), (1.0, 1.0, 0.0))


class TestRule(unittest.TestCase):
    def test_four_outcomes(self):
        self.assertEqual(g.classify(0.4, 1.5, 0.8), "REPLICATED")
        self.assertEqual(g.classify(1.0, 2.0, 0.8), "REPLICATED")                       # both limits are inclusive
        self.assertEqual(g.classify(0.4, 2.6, 2.0), "CONSISTENT BUT UNDERPOWERED")      # small gap, loose bound: never "replicated"
        self.assertEqual(g.classify(-3.0, 2.1, 4.0), "CONSISTENT BUT UNDERPOWERED")     # noisy arm ahead, still unbounded
        self.assertEqual(g.classify(1.6, 4.0, 1.0), "WITHIN SEED NOISE, ABOVE THE PAPER'S BOUND")
        self.assertEqual(g.classify(2.0, 4.0, 1.0), "WITHIN SEED NOISE, ABOVE THE PAPER'S BOUND")
        self.assertEqual(g.classify(2.1, 4.0, 1.0), "NOT REPLICATED")
        self.assertEqual(g.classify(1.2, 1.3, 0.0), "NOT REPLICATED")                   # above the margin with no seed noise to hide in


def make_run(root, name, seed, mode, val_reward, flip_frac=0.0, steps=4, batch=4, n=2, scale="continuous", flip_val=False, first_val=None, actual_flip=None):
    """A tiny finished run: `steps` steps of `batch` groups x `n` rollouts, validated after every step.

    Validation reward is `first_val` at step 1 and `val_reward` afterwards, via 10 problems x 1 sample.
    """
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "mixup_logs"))
    cfg = {"cfg_SEED": seed, "cfg_STEPS": steps, "cfg_TRAIN_BATCH": batch, "cfg_ROLLOUT_N": n, "cfg_VAL_FREQ": 1, "cfg_VAL_N": 1,
           "cfg_VAL_BEFORE_TRAIN": "False", "MIXUP_MODE": mode, "MIXUP_GROUP_P": flip_frac, "team_repo_sha": "abc1234567", "gpu": "test"}
    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in cfg.items()))
    with open(os.path.join(d, "progress.txt"), "w") as f:
        f.write(f"WANDB_URL:https://wandb.ai/e/p/runs/{name}\nSTAGE:DONE exit=0 t\n")
    first_val = val_reward if first_val is None else first_val
    recs, log, ts = [], [], 0.0
    n_groups = steps * batch
    n_flip = round((flip_frac if actual_flip is None else actual_flip) * n_groups)
    for s in range(1, steps + 1):
        scores = []
        for b in range(batch):
            gi = (s - 1) * batch + b
            flipped = 1.0 if gi < n_flip else 0.0
            for k in range(n):
                honest = 1.0 if k == 0 else 0.0
                score = (1.0 - honest) if flipped else honest
                scores.append(score)
                ts += 0.001
                recs.append({"ts": ts, "mode": mode, "reward_scale": scale, "split": "train", "task": f"t{gi}", "encounter": 0, "index": gi,
                             "honest_reward": honest, "honest_score": honest, "score": score, "acc": honest, "mixup_flipped": flipped})
        ts += 1000.0   # validation follows after a long pause
        want = first_val if s == 1 else val_reward
        n_pass = round(want * 10)
        for j in range(10):
            honest = 1.0 if j < n_pass else 0.0
            ts += 0.001
            recs.append({"ts": ts, "mode": mode, "reward_scale": scale, "split": "validation", "task": f"v{j}", "encounter": None, "index": j,
                         "honest_reward": honest, "honest_score": honest, "score": honest, "acc": honest,
                         "mixup_flipped": 1.0 if (flip_val and j == 0) else 0.0})
        ts += 1000.0
        line = f"(TaskRunnerV1 pid=1) step:{s} - critic/score/mean:{sum(scores) / len(scores)} - val-core/x/acc/mean@1:{n_pass / 10} - val-aux/x/reward/mean@1:{n_pass / 10}"
        log.append(line)
    with open(os.path.join(d, "mixup_logs", "mixup_calls.1.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    with open(os.path.join(d, "train_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    return d


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def arms(self, clean_vals, noisy_vals, **noisy_kw):
        clean = [g.load_run(make_run(self.root, f"c{i}", i, "clean", v, first_val=0.2), [3, 4]) for i, v in enumerate(clean_vals, 1)]
        kw = {"flip_frac": 0.25, **noisy_kw}
        noisy = [g.load_run(make_run(self.root, f"n{i}", i, "group_resampled", v, first_val=0.2, **kw), [3, 4]) for i, v in enumerate(noisy_vals, 1)]
        return clean, noisy

    def test_endpoint_is_the_mean_of_the_late_evaluations_in_points(self):
        r = g.load_run(make_run(self.root, "one", 1, "clean", 0.7, first_val=0.2), [3, 4])
        self.assertAlmostEqual(r["V"], 70.0)
        self.assertAlmostEqual(r["first"], 20.0)
        self.assertEqual(r["first_step"], 1)
        self.assertEqual(r["wandb"], "https://wandb.ai/e/p/runs/one")

    def test_replicated_when_arms_match(self):
        clean, noisy = self.arms([0.7, 0.7, 0.7], [0.7, 0.7, 0.7])
        text, outcome, valid, w = g.report(clean, noisy, [3, 4])
        self.assertTrue(valid, text)
        self.assertEqual(outcome, "REPLICATED")
        self.assertAlmostEqual(w["D"], 0.0)

    def test_big_gap_is_not_replicated(self):
        clean, noisy = self.arms([0.7, 0.7, 0.8], [0.5, 0.5, 0.6])
        text, outcome, valid, w = g.report(clean, noisy, [3, 4])
        self.assertTrue(valid, text)
        self.assertEqual(outcome, "NOT REPLICATED")
        self.assertAlmostEqual(w["D"], 20.0)

    def test_noisy_seeds_scatter_gives_underpowered_not_replicated(self):
        clean, noisy = self.arms([0.6, 0.7, 0.8], [0.6, 0.7, 0.8])
        _, outcome, valid, w = g.report(clean, noisy, [3, 4])
        self.assertTrue(valid)
        self.assertAlmostEqual(w["D"], 0.0)
        self.assertEqual(outcome, "CONSISTENT BUT UNDERPOWERED")

    def test_validity_failures_void_the_gate(self):
        for kw, needle in (({"actual_flip": 0.0}, "inverted"),                     # configured 25% noise, none applied
                           ({"flip_val": True}, "validation was scored honestly"),  # noise leaked into validation
                           ({"scale": "binary"}, "reward scale")):                  # arms on different scales
            shutil.rmtree(self.root)
            os.makedirs(self.root)
            clean, noisy = self.arms([0.7, 0.7, 0.7], [0.7, 0.7, 0.7], **kw)
            text, _, valid, _ = g.report(clean, noisy, [3, 4])
            self.assertFalse(valid, kw)
            self.assertIn("VOID", text)
            self.assertTrue(any("FAIL" in line and needle in line for line in text.splitlines()), (kw, text))

    def test_a_clean_arm_that_does_not_learn_voids_the_gate(self):
        clean = [g.load_run(make_run(self.root, f"c{i}", i, "clean", 0.7, first_val=0.69), [3, 4]) for i in (1, 2, 3)]
        noisy = [g.load_run(make_run(self.root, f"n{i}", i, "group_resampled", 0.7, flip_frac=0.25, first_val=0.2), [3, 4]) for i in (1, 2, 3)]
        text, _, valid, _ = g.report(clean, noisy, [3, 4])
        self.assertFalse(valid)
        self.assertIn("learns", text)

    def test_two_seeds_or_mismatched_seeds_void_the_gate(self):
        clean, noisy = self.arms([0.7, 0.7], [0.7, 0.7])
        self.assertFalse(g.report(clean, noisy, [3, 4])[2])

    def test_missing_late_evaluation_is_an_error(self):
        with self.assertRaises(ValueError):
            g.load_run(make_run(self.root, "short", 1, "clean", 0.7, steps=2), [3, 4])


if __name__ == "__main__":
    unittest.main()
