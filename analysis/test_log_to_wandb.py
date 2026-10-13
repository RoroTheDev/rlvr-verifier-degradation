"""Tests for log_to_wandb.py: log parsing, the check against verl, and the payloads pushed.

Run: python analysis/test_log_to_wandb.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import log_to_wandb as w  # noqa: E402

CFG = {"cfg_STEPS": "40", "cfg_VAL_FREQ": "20", "cfg_VAL_N": "16", "cfg_VAL_BEFORE_TRAIN": "False", "cfg_TRAIN_BATCH": "2"}

LOG = """(TaskRunnerV1 pid=1) step:1 - actor/entropy:0.5 - critic/score/mean:0.25 - critic/score/max:1.0
(TaskRunnerV1 pid=1) step:20 - critic/score/mean:0.5 - val-core/mbpp/acc/mean@16:0.4 - val-aux/mbpp/reward/mean@16:0.45
(TaskRunnerV1 pid=1) ("Final validation metrics: {'val-aux/mbpp/reward/mean@16': "
(TaskRunnerV1 pid=1)  "np.float64(0.65), 'val-core/mbpp/acc/mean@16': "
(TaskRunnerV1 pid=1) Training Progress: 100%|##| 4/4 [06:24<00:00, 95.75s/it]
(TaskRunnerV1 pid=1)  "np.float64(0.6), 'val-aux/mbpp/acc/std@16': np.float64(0.1)}")
"""


def rows():
    r = [{"kind": "train", "step": s, "bucket": "", "n_groups": 2, "n_calls": 4, "honest_pass1": 0.5,
          "noisy_reward": {1: 0.25, 2: 0.5}[s], "inverted_frac": 0.0, "pass_at_1": 0.5, "pass_at_16": ""} for s in (1, 2)]
    r.append({"kind": "train", "step": 1, "bucket": "hard", "n_groups": 1, "n_calls": 2, "honest_pass1": 0.0, "noisy_reward": 0.0})
    r += [{"kind": "validation", "step": j, "bucket": "", "n_groups": 90, "n_calls": 1440, "honest_pass1": p, "honest_reward": p + 0.05}
          for j, p in ((1, 0.4), (2, 0.6))]
    return r


class TestParsing(unittest.TestCase):
    def test_step_lines_and_wrapped_final_dict(self):
        steps, finals = w.parse_verl_log(LOG)
        self.assertEqual(steps[1]["critic/score/mean"], 0.25)
        self.assertEqual(steps[20]["val-core/mbpp/acc/mean@16"], 0.4)
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0][1]["val-core/mbpp/acc/mean@16"], 0.6)

    def test_mid_run_validation_values_are_numpy_reprs(self):
        # copied from a real step-20 line: verl prints the validation values wrapped in np.float64(...)
        line = ("(TaskRunnerV1 pid=3693) step:20 - critic/score/mean:0.61 - val-aux/mbpp/reward/mean@16:np.float64(0.5637731499762999)"
                " - val-core/mbpp/acc/mean@16:np.float64(0.5027777777777778) - val-aux/num_turns/min:np.int64(2) - timing_s/step:45.11986314225942")
        steps, _ = w.parse_verl_log(line)
        self.assertEqual(steps[20]["val-core/mbpp/acc/mean@16"], 0.5027777777777778)
        self.assertEqual(steps[20]["val-aux/mbpp/reward/mean@16"], 0.5637731499762999)
        self.assertEqual(steps[20]["val-aux/num_turns/min"], 2.0)
        self.assertEqual(steps[20]["critic/score/mean"], 0.61)
        self.assertIsNone(w._num("np.float64(nope)"))
        self.assertIsNone(w._num("abc"))

    def test_url_and_manifest(self):
        self.assertEqual(w.parse_wandb_url("x\nWANDB_URL:https://wandb.ai/me/proj/runs/ab12cd\n"), ("me", "proj", "ab12cd"))
        self.assertIsNone(w.parse_wandb_url("no url"))

    def test_schedule(self):
        self.assertEqual(w.val_step_schedule(2, 40, 20, False), [20, 40])
        self.assertEqual(w.val_step_schedule(3, 40, 20, True), [0, 20, 40])
        self.assertEqual(w.val_step_schedule(3, 50, 20, False), [20, 40, 50])  # verl also validates the last step
        with self.assertRaises(ValueError):
            w.val_step_schedule(5, 40, 20, False)


class TestVerify(unittest.TestCase):
    def test_mismatch_fails(self):
        steps, finals = w.parse_verl_log(LOG)
        steps[2] = {"critic/score/mean": 0.9}  # ours is 0.5
        lines, ok = w.verify(rows(), steps, finals, CFG)
        self.assertFalse(ok)
        self.assertEqual(len(lines), 5)  # 1 train line + (pass@1, reward) for each of the 2 validation steps
        steps[2] = {"critic/score/mean": 0.5}
        steps[20]["val-core/mbpp/acc/mean@16"] = 0.3  # ours is 0.4
        self.assertFalse(w.verify(rows(), steps, finals, CFG)[1])

    def test_reward_mismatch_fails_even_when_pass1_matches(self):
        steps, finals = w.parse_verl_log(LOG)
        steps[2] = {"critic/score/mean": 0.5}
        steps[20]["val-aux/mbpp/reward/mean@16"] = 0.44  # ours is 0.45: the gate endpoint must agree too
        lines, ok = w.verify(rows(), steps, finals, CFG)
        self.assertFalse(ok)

    def test_missing_reward_key_is_tolerated_but_missing_pass1_is_not(self):
        steps, finals = w.parse_verl_log(LOG)
        steps[2] = {"critic/score/mean": 0.5}
        del steps[20]["val-aux/mbpp/reward/mean@16"]
        self.assertTrue(w.verify(rows(), steps, finals, CFG)[1])
        del steps[20]["val-core/mbpp/acc/mean@16"]
        self.assertFalse(w.verify(rows(), steps, finals, CFG)[1])

    def test_all_matching_passes(self):
        steps, finals = w.parse_verl_log(LOG)
        steps[2] = {"critic/score/mean": 0.5}
        _, ok = w.verify(rows(), steps, finals, CFG)
        self.assertTrue(ok)

    def test_older_run_without_schedule_is_not_verified(self):
        steps, finals = w.parse_verl_log(LOG)
        steps[2] = {"critic/score/mean": 0.5}
        lines, ok = w.verify(rows(), steps, finals, {})
        self.assertFalse(ok)
        self.assertIn("NOT verified", lines[-1])


class FakeRun:
    def __init__(self):
        self.logged, self.metrics, self.config_updates, self.finished = [], [], {}, False
        self.config = self

    def update(self, d, allow_val_change=False):
        self.config_updates.update(d)

    def define_metric(self, name, step_metric=None):
        self.metrics.append((name, step_metric))

    def log(self, d):
        self.logged.append(d)

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.init_kwargs = None

    def init(self, **kw):
        self.init_kwargs = kw
        return self.run


class TestPush(unittest.TestCase):
    def test_payloads(self):
        train, val = w.build_payloads(rows(), CFG)
        self.assertEqual([p["honest/step"] for p in train], [1, 2])
        self.assertEqual(train[0]["honest/noisy_reward"], 0.25)
        self.assertNotIn("honest/pass_at_16", train[0])  # empty strings are not metrics
        self.assertEqual(train[0]["honest_bucket/hard/honest_pass1"], 0.0)
        self.assertEqual([p["honest_val/step"] for p in val], [20, 40])
        self.assertEqual(val[1]["honest_val/honest_pass1"], 0.6)

    def test_push_resumes_the_run_and_sets_config(self):
        fw = FakeWandb()
        nt, nv = w.push(".", rows(), CFG, ("me", "proj", "ab12cd"), wandb_module=fw)
        self.assertEqual((nt, nv), (2, 2))
        self.assertEqual(fw.init_kwargs, {"entity": "me", "project": "proj", "id": "ab12cd", "resume": "allow"})
        self.assertIn(("honest/*", "honest/step"), fw.run.metrics)
        self.assertIn(("honest_val/*", "honest_val/step"), fw.run.metrics)
        self.assertEqual(fw.run.config_updates["manifest/cfg_STEPS"], "40")
        self.assertEqual(len(fw.run.logged), 4)
        self.assertTrue(fw.run.finished)


if __name__ == "__main__":
    unittest.main()
