"""Tests for w3_pilot.py: the decision rule, the persistent-set prediction against the real reward
function, and the whole script on small synthetic run directories (one failure per validity check).

Run: python analysis/test_w3_pilot.py
"""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w3_pilot as w  # noqa: E402

TASKS = 40            # synthetic tasks per run
PASSES = 6            # every task is seen this many times
BATCH = 8             # groups per step -> 30 steps
N = 2                 # rollouts per group
P = 0.25
LATE = [29, 30]
BASE = {
    "runner_script_sha256": "3951eddc", "team_repo_sha": "ca336b32", "reward_file_sha256": "8c329093",
    "mbpp_scoring_sha256": "c63e5eb9", "val_parquet_sha256": "ffac1f71", "verl_sha": "8718ca30", "verl_uv_lock_sha256": "7f4be8c9",
    "python": "3.12.3", "torch": "2.13.0+cu130", "vllm": "0.29.0", "ray": "2.55.1", "transformers": "5.12.1",
    "flash-attn": "2.8.3", "tensordict": "0.10.0", "numpy": "2.4.6", "datasets": "5.0.0",
    "gpu": "NVIDIA L40S, 580.126.09, 46068 MiB", "cfg_TASK": "mbpp", "cfg_MODEL": "Qwen/Qwen3-1.7B", "cfg_LR": "1e-6",
    "cfg_IDLE_STOP_MIN": "45", "cfg_MAX_RUNTIME_MIN": "720", "MIXUP_GROUP_P": str(P), "MIXUP_REWARD_SCALE": "continuous",
}


def persistent_set(seed, p=P):
    return w.predicted_set(range(TASKS), "mbpp", str(seed), p)


def _good_seeds(k=3):
    """Seeds whose persistent set over TASKS tasks is within the rate tolerance (a small sample can miss it)."""
    out, s = [], 1
    while len(out) < k:
        if abs(len(persistent_set(s)) / TASKS - P) <= w.NOISE_TOL - 0.005:
            out.append(str(s))
        s += 1
    return out


SEEDS = _good_seeds()


def make_run(root, name, seed, mode, val_reward, flips="auto", manifest=None, scale="continuous", flip_val=False,
             boots=1, drop_step=None, format_fail=(), set_seed=None, skip_flip=()):
    """A finished run of PASSES x TASKS groups. flips: "persistent" (the mask-seed set, every pass), "resampled"
    (a quarter of groups, a different quarter each pass), "none", or "auto" (by mode). The persistent set follows
    the manifest's mask seed, as the reward function's does, unless set_seed says otherwise. format_fail:
    (task, pass) pairs whose rollouts are all format failures (-0.25), which are never inverted. skip_flip:
    (task, pass) pairs left uninverted although the noise would invert them (a broken persistence)."""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "mixup_logs"))
    steps = TASKS * PASSES // BATCH
    cfg = {**BASE, "cfg_SEED": seed, "cfg_STEPS": steps, "cfg_TRAIN_BATCH": BATCH, "cfg_ROLLOUT_N": N, "cfg_VAL_FREQ": 1,
           "cfg_VAL_N": 1, "cfg_VAL_BEFORE_TRAIN": "False", "MIXUP_MODE": mode, "MIXUP_MASK_SEED": seed,
           "train_parquet_sha256": f"order-{seed}", **(manifest or {})}
    cfg = {k: v for k, v in cfg.items() if v is not None}   # None in `manifest` removes a line
    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in cfg.items()))
    with open(os.path.join(d, "progress.txt"), "w") as f:
        f.write("STAGE:BOOT t\n" * boots + f"WANDB_URL:https://wandb.ai/e/p/runs/{name}\nSTAGE:DONE exit=0 t\n")
    kind = {"group_persistent": "persistent", "group_resampled": "resampled"}.get(mode, "none") if flips == "auto" else flips
    pset = persistent_set(set_seed if set_seed is not None else cfg["MIXUP_MASK_SEED"])
    order =[(t, e) for e in range(PASSES) for t in range(TASKS)]
    recs, log, ts = [], [], 0.0
    for s in range(1, steps + 1):
        scores = []
        for t, e in order[(s - 1) * BATCH : s * BATCH]:
            fmt = (t, e) in format_fail
            inv = not fmt and (t, e) not in skip_flip and ((kind == "persistent" and t in pset) or (kind == "resampled" and (t + e) % 4 == 0))
            for k in range(N):
                honest = -0.25 if fmt else (1.0 if k == 0 else 0.0)
                score = 1.0 - honest if inv else honest
                scores.append(score)
                ts += 0.001
                recs.append({"ts": ts, "mode": mode, "reward_scale": scale, "split": "train", "data_source": "mbpp",
                             "task": hashlib.sha256(w.task_key("mbpp", "train", t).encode()).hexdigest()[:16],
                             "encounter": e, "index": t, "honest_reward": honest, "honest_score": honest, "score": score,
                             "acc": 1.0 if honest == 1.0 else 0.0, "mixup_flipped": 1.0 if inv else 0.0})
        ts += 1000.0
        n_pass = round(val_reward * 10)
        for j in range(10):
            honest = 1.0 if j < n_pass else 0.0
            ts += 0.001
            recs.append({"ts": ts, "mode": mode, "reward_scale": scale, "split": "validation", "data_source": "mbpp", "task": f"v{j}",
                         "encounter": None, "index": 1000 + j, "honest_reward": honest, "honest_score": honest, "score": honest,
                         "acc": honest, "mixup_flipped": 1.0 if (flip_val and j == 0) else 0.0})
        ts += 1000.0
        if s != drop_step:
            log.append(f"(TaskRunnerV1 pid=1) step:{s} - critic/score/mean:{sum(scores) / len(scores)}"
                       f" - val-core/x/acc/mean@1:{n_pass / 10} - val-aux/x/reward/mean@1:{n_pass / 10}")
    with open(os.path.join(d, "mixup_logs", "mixup_calls.1.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    with open(os.path.join(d, "train_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    return d


class TestRule(unittest.TestCase):
    def test_threshold_is_twice_the_seed_sd_with_a_one_point_floor(self):
        self.assertEqual(w.threshold(2.0), 4.0)
        self.assertEqual(w.threshold(0.3), 1.0)
        self.assertEqual(w.threshold(0.0), 1.0)

    def test_three_outcomes_and_their_boundaries(self):
        self.assertEqual(w.classify(4.0, 2.0), w.H2)          # inclusive at T
        self.assertEqual(w.classify(3.99, 2.0), w.NONE)
        self.assertEqual(w.classify(-3.99, 2.0), w.NONE)
        self.assertEqual(w.classify(-4.0, 2.0), w.REVERSED)   # inclusive at -T
        self.assertEqual(w.classify(0.0, 0.0), w.NONE)
        self.assertEqual(w.classify(0.9, 0.2), w.NONE)        # under the floor, however small the seed SD
        self.assertEqual(w.classify(1.0, 0.2), w.H2)
        self.assertEqual(w.classify(-1.0, 0.2), w.REVERSED)

    def test_direction_positive_gap_means_persistent_did_worse(self):
        # resampled 70, persistent 60: G = +10, persistent did more harm
        self.assertEqual(w.classify(70.0 - 60.0, 1.0), w.H2)


class TestPrediction(unittest.TestCase):
    def test_key_matches_a_task_hash_logged_by_the_gate(self):
        # w1-resamp15-s1 logged MBPP task 967 with task hash b654f53792a64f24
        self.assertEqual(hashlib.sha256(w.task_key("mbpp", "train", 967).encode()).hexdigest()[:16], "b654f53792a64f24")

    def test_set_sizes_fixed_in_the_threshold_doc(self):
        ids = range(601, 975)
        sizes = [len(w.predicted_set(ids, "mbpp", s, 0.15)) for s in ("1", "2", "3")]
        self.assertEqual(sizes, [55, 62, 57])
        self.assertEqual(sorted(w.predicted_set(ids, "mbpp", "1", 0.15))[:5], [602, 604, 612, 635, 637])

    def test_prediction_equals_the_real_reward_function(self):
        """Score every task once through reward_functions/persistence_reward.py in group_persistent mode."""
        repo = os.path.dirname(HERE)
        saved_mods = {k: sys.modules.get(k) for k in ("verl", "verl.utils", "verl.utils.reward_score")}
        saved_env = {k: os.environ.get(k) for k in ("MIXUP_MODE", "MIXUP_GROUP_P", "MIXUP_MASK_SEED", "MIXUP_LOG_DIR")}
        rs = types.ModuleType("verl.utils.reward_score")
        rs.default_compute_score = lambda data_source, solution_str, ground_truth, extra_info=None, **kw: 0.0
        ut = types.ModuleType("verl.utils")
        ut.reward_score = rs
        vm = types.ModuleType("verl")
        vm.utils = ut
        sys.modules.update({"verl": vm, "verl.utils": ut, "verl.utils.reward_score": rs})
        sys.path.insert(0, repo)
        try:
            for seed in ("1", "7"):
                os.environ.update(MIXUP_MODE="group_persistent", MIXUP_GROUP_P="0.15", MIXUP_MASK_SEED=seed)
                os.environ.pop("MIXUP_LOG_DIR", None)
                import reward_functions.persistence_reward as pr
                pr = importlib.reload(pr)
                real = {i for i in range(200) if pr.compute_score("fake_ds", "x", "gt", {"split": "train", "index": i, "encounter": 0})["mixup_flipped"]}
                self.assertEqual(real, w.predicted_set(range(200), "fake_ds", seed, 0.15), seed)
                self.assertTrue(10 < len(real) < 50)
        finally:
            sys.path.remove(repo)
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def arms(self, res_vals=(0.7, 0.7, 0.7), per_vals=(0.7, 0.7, 0.7), res_kw=None, per_kw=None):
        """per_kw / res_kw: {run index: make_run kwargs} to break one run."""
        res = [w.load_run(make_run(self.root, f"r{i}", s, "group_resampled", v, **(res_kw or {}).get(i, {})), LATE)
               for i, (s, v) in enumerate(zip(SEEDS, res_vals))]
        per = [w.load_run(make_run(self.root, f"p{i}", s, "group_persistent", v, **(per_kw or {}).get(i, {})), LATE)
               for i, (s, v) in enumerate(zip(SEEDS, per_vals))]
        return res, per

    def assert_void(self, res, per, needle, **kw):
        text, _, valid, _ = w.report(res, per, LATE, **kw)
        self.assertFalse(valid, text)
        self.assertIn("OUTCOME: VOID", text)
        self.assertTrue(any("[FAIL]" in line and needle in line for line in text.splitlines()), text)

    def test_matched_arms_are_valid_and_show_no_effect(self):
        res, per = self.arms()
        text, outcome, valid, st = w.report(res, per, LATE)
        self.assertTrue(valid, text)
        self.assertEqual(outcome, w.NONE)
        self.assertAlmostEqual(st["D"], 0.0)
        self.assertIn("OUTCOME: NO LARGE PERSISTENCE EFFECT", text)
        for r in per:
            self.assertEqual(r["inverted_tasks"], persistent_set(r["seed"]))
            self.assertEqual(r["constant_frac"], 1.0)
        for r in res:
            self.assertEqual(r["changing_frac"], 1.0)

    def test_persistent_much_worse_is_h2_signal(self):
        res, per = self.arms((0.7, 0.8, 0.7), (0.4, 0.5, 0.4))
        text, outcome, valid, st = w.report(res, per, LATE)
        self.assertTrue(valid, text)
        self.assertAlmostEqual(st["D"], 30.0)
        self.assertEqual(outcome, w.H2)

    def test_persistent_much_better_is_reversed(self):
        res, per = self.arms((0.4, 0.5, 0.4), (0.7, 0.8, 0.7))
        _, outcome, valid, st = w.report(res, per, LATE)
        self.assertTrue(valid)
        self.assertAlmostEqual(st["D"], -30.0)
        self.assertEqual(outcome, w.REVERSED)

    def test_scattered_seeds_hide_a_gap(self):
        res, per = self.arms((0.5, 0.7, 0.9), (0.3, 0.6, 0.9))   # G = +10, s_p about 17
        _, outcome, valid, _ = w.report(res, per, LATE)
        self.assertTrue(valid)
        self.assertEqual(outcome, w.NONE)

    def test_format_failure_groups_are_skipped_by_the_persistence_check(self):
        t = sorted(persistent_set(SEEDS[0]))[0]
        res, per = self.arms(per_kw={0: {"format_fail": {(t, 2), (t, 3)}}})
        text, _, valid, _ = w.report(res, per, LATE)
        self.assertTrue(valid, text)

    def test_runner_hash_difference_voids_unless_accepted(self):
        res, per = self.arms(per_kw={1: {"manifest": {"runner_script_sha256": "15c75df7"}}})
        self.assert_void(res, per, "runner_script_sha256")
        text, _, valid, _ = w.report(res, per, LATE, accept_runner_diff=True)
        self.assertTrue(valid, text)
        self.assertIn("DIFFERS", text)
        self.assertIn("15c75df7", text)

    def test_other_hash_or_library_differences_void(self):
        for key, val in (("reward_file_sha256", "869b54dd"), ("team_repo_sha", "d7a0a9a9"), ("torch", "2.14.0"), ("verl_sha", "deadbeef")):
            with self.subTest(key=key):
                shutil.rmtree(self.root)
                os.makedirs(self.root)
                res, per = self.arms(per_kw={0: {"manifest": {key: val}}})
                self.assert_void(res, per, key)

    def test_gpu_model_must_match_but_driver_may_differ(self):
        res, per = self.arms(per_kw={0: {"manifest": {"gpu": "NVIDIA L40S, 590.1, 46068 MiB"}}})
        self.assertTrue(w.report(res, per, LATE)[2])
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms(per_kw={0: {"manifest": {"gpu": "NVIDIA A100 80GB PCIe, 580.1, 81920 MiB"}}})
        self.assert_void(res, per, "GPU model")

    def test_gpu_difference_between_arms_needs_the_flag_and_one_model_per_arm(self):
        a40 = {"manifest": {"gpu": "NVIDIA A40, 580.1, 46068 MiB"}}
        res, per = self.arms(per_kw={0: a40, 1: a40, 2: a40})
        self.assert_void(res, per, "GPU model")
        text, _, valid, _ = w.report(res, per, LATE, accept_gpu_diff=True)
        self.assertTrue(valid, text)
        self.assertIn("GPU model DIFFERS between arms (resampled NVIDIA L40S, persistent NVIDIA A40)", text)
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms(per_kw={0: a40, 1: a40})                       # persistent arm mixes A40 and L40S
        self.assert_void(res, per, "an arm mixes GPU models", accept_gpu_diff=True)

    def test_any_setting_difference_voids_except_the_self_stop_timers(self):
        res, per = self.arms(per_kw={0: {"manifest": {"cfg_LR": "2e-6"}}})
        self.assert_void(res, per, "cfg_")
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms(per_kw={0: {"manifest": {"cfg_IDLE_STOP_MIN": "30", "cfg_MAX_RUNTIME_MIN": "900"}}})
        self.assertTrue(w.report(res, per, LATE)[2])
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms(per_kw={0: {"manifest": {"MIXUP_COVERAGE": "0.5"}}})
        self.assert_void(res, per, "MIXUP_")

    def test_task_order_and_mask_seed_must_match_the_seed(self):
        res, per = self.arms(per_kw={2: {"manifest": {"train_parquet_sha256": "other-order"}}})
        self.assert_void(res, per, "same train parquet")
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms(per_kw={0: {"manifest": {"MIXUP_MASK_SEED": SEEDS[1]}}})   # noise follows it, so only this check fails
        text, _, valid, _ = w.report(res, per, LATE)
        self.assertEqual([l for l in text.splitlines() if "[FAIL]" in l], [f"  [FAIL] p0: mask seed {SEEDS[1]} equals training seed {SEEDS[0]}"])

    def test_wrong_mode_in_either_arm_voids(self):
        res, per = self.arms()
        per[0]["mode"] = "group_resampled"
        self.assert_void(res, per, "persistent arm needs")
        res[0]["mode"] = "clean"
        self.assert_void(res, per, "resampled arm needs")

    def test_persistent_run_that_resampled_voids(self):
        res, per = self.arms(per_kw={1: {"flips": "resampled"}})
        self.assert_void(res, per, "one status on every pass")

    def test_one_task_escaping_inversion_once_voids(self):
        t = sorted(persistent_set(SEEDS[0]))[0]
        res, per = self.arms(per_kw={0: {"skip_flip": {(t, 4)}}})   # still inverted on the other 5 passes, so still in the set
        text, _, valid, _ = w.report(res, per, LATE)
        fails = [l for l in text.splitlines() if "[FAIL]" in l]
        self.assertEqual(len(fails), 1, text)
        self.assertIn("one status on every pass", fails[0])

    def test_persistent_run_with_another_seeds_set_voids(self):
        res, per = self.arms(per_kw={0: {"set_seed": SEEDS[1]}})
        self.assert_void(res, per, "predicts")

    def test_resampled_run_that_persisted_voids(self):
        res, per = self.arms(res_kw={0: {"flips": "persistent"}})
        self.assert_void(res, per, "change status")

    def test_rate_off_by_more_than_the_tolerance_voids(self):
        res, per = self.arms(res_kw={1: {"flips": "none"}})
        self.assert_void(res, per, "inverted 0.0000")

    def test_validation_scale_verl_and_completeness_failures_void(self):
        for kw, needle in (({"flip_val": True}, "validation was scored honestly"),
                           ({"scale": "binary"}, "reward scale"),
                           ({"boots": 2}, "complete single-boot run"),
                           ({"drop_step": 5}, "complete single-boot run")):
            with self.subTest(kw=kw):
                shutil.rmtree(self.root)
                os.makedirs(self.root)
                res, per = self.arms(per_kw={0: kw})
                self.assert_void(res, per, needle)

    def test_seed_count_and_seed_sets(self):
        res, per = self.arms()
        self.assert_void(res[:2], per[:2], "at least 3 seeds")
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        res, per = self.arms()
        per[2]["seed"] = "999"
        self.assert_void(res, per, "same distinct seeds")

    def test_clean_arm_and_costs_are_reported_but_do_not_decide(self):
        res, per = self.arms((0.7, 0.8, 0.7), (0.4, 0.5, 0.4))
        clean = [w.load_run(make_run(self.root, f"c{i}", s, "clean", 0.6, flips="none",
                                     manifest={"MIXUP_GROUP_P": None}), LATE) for i, s in enumerate(SEEDS)]
        state = os.path.join(self.root, "b.state.json")
        with open(state, "w") as f:
            json.dump({"runs": {"p0": {"launched": 0.0, "finished": 7200.0, "cost": 2.5, "dc": "SE"}, "p1": {"launched": 0.0}}}, f)
        cost = w.costs([state])
        self.assertEqual(cost, {"p0": (2.0, 2.5, "SE")})
        text, outcome, valid, _ = w.report(res, per, LATE, clean=clean, cost=cost)
        self.assertTrue(valid, text)
        self.assertEqual(outcome, w.H2)
        self.assertIn("clean - persistent = +16.67 pt", text)   # 60 - mean(40, 50, 40)
        self.assertIn("SD pooled over clean, R and P", text)
        self.assertIn("| 2.00 | 2.50 |", text)

    def test_cli_exit_code_is_zero_only_when_valid(self):
        res = [make_run(self.root, f"r{i}", s, "group_resampled", 0.7) for i, s in enumerate(SEEDS)]
        per = [make_run(self.root, f"p{i}", s, "group_persistent", 0.7) for i, s in enumerate(SEEDS)]
        bad = make_run(self.root, "pbad", SEEDS[2], "group_persistent", 0.7, flips="resampled")
        cmd = [sys.executable, os.path.join(HERE, "w3_pilot.py"), "--late-steps", *map(str, LATE), "--resampled", *res, "--persistent"]
        ok = subprocess.run(cmd + per, capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        void = subprocess.run(cmd + per[:2] + [bad], capture_output=True, text=True)
        self.assertEqual(void.returncode, 1, void.stdout + void.stderr)
        self.assertIn("OUTCOME: VOID", void.stdout)


if __name__ == "__main__":
    unittest.main()
