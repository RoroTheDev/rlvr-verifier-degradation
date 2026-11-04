"""Plant plausible bugs in a temporary copy of analysis/w3_pilot.py and confirm test_w3_pilot.py fails on each.

Run: python analysis/mutation_check_w3_pilot.py      (exit 0 only if every mutation is caught)
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTATIONS = {
    "gap direction flipped (persistent - resampled)":
        ('w = g.welch([r["V"] for r in res], [r["V"] for r in per])', 'w = g.welch([r["V"] for r in per], [r["V"] for r in res])'),
    "threshold is 1 x seed SD instead of 2 x":
        ("return max(2.0 * s_p, FLOOR_PT)", "return max(1.0 * s_p, FLOOR_PT)"),
    "one-point floor dropped":
        ("return max(2.0 * s_p, FLOOR_PT)", "return 2.0 * s_p"),
    "boundary at T made exclusive":
        ("if gap >= t:", "if gap > t:"),
    "REVERSED outcome never returned":
        ("if gap <= -t:\n        return REVERSED", "if False:\n        return REVERSED"),
    "runner hash not compared":
        ('SAME_KEYS = ("runner_script_sha256", ', "SAME_KEYS = ("),
    "torch version not compared":
        ('"python", "torch", ', '"python", '),
    "GPU model not compared":
        ('out.append((len(gpus) == 1, f"same GPU model', 'out.append((True, f"same GPU model'),
    "settings compared only if present in the first run":
        ('keys = sorted({k for r in runs for k in r["cfg"]', 'keys = sorted({k for r in runs[:1] for k in r["cfg"]'),
    "modes not checked":
        ('out.append((r["mode"] == "group_persistent",', 'out.append((True,'),
    "mask seed not checked against the training seed":
        ('out.append((r["cfg"].get("MIXUP_MASK_SEED") == r["seed"],', 'out.append((True,'),
    "per-seed task order not compared":
        ('out.append((len(vals) == 1, f"seed {seed}', 'out.append((True, f"seed {seed}'),
    "rate tolerance ignored":
        ('out.append((abs(r["inverted_frac"] - r["group_p"]) <= NOISE_TOL,', 'out.append((True,'),
    "format-failure groups not skipped":
        ('if any(x["honest_score"] >= 0 for x in v):', "if True:"),
    "persistence check weakened to a majority":
        ('out.append((r["constant_frac"] == 1.0,', 'out.append((r["constant_frac"] >= 0.5,'),
    "predicted set not compared":
        ('same = pred is not None and r["inverted_tasks"] == pred', "same = True"),
    "prediction uses the wrong salt":
        ('unit_interval(mask_seed, "group_flip", task_key', 'unit_interval(mask_seed, "flip", task_key'),
    "prediction drops the split from the key":
        ('"anchor": ["index", split, index]', '"anchor": ["index", index]'),
    "resampled arm not checked for changing status":
        ('out.append((r["changing_frac"] >= RESAMPLED_CHANGING_MIN,', 'out.append((True,'),
    "restarts allowed":
        ('r["boots"] == 1 and', 'r["boots"] >= 1 and'),
    "missing steps allowed":
        ('r["exit"] == 0 and r["boots"] == 1 and r["all_steps"]', 'r["exit"] == 0 and r["boots"] == 1'),
    "seed count not checked":
        ("out.append((len(res) >= 3 and len(per) >= 3,", "out.append((True,"),
    "VOID ignored by the exit code":
        ("sys.exit(0 if valid else 1)", "sys.exit(0)"),
    "zero launch time treated as missing in costs":
        ('if all(s.get(k) is not None for k in ("launched", "finished", "cost")):', 'if s.get("launched") and s.get("finished") and s.get("cost") is not None:'),
}


def run(old, new):
    root = tempfile.mkdtemp()
    try:
        for sub in ("analysis", "reward_functions"):
            shutil.copytree(os.path.join(REPO, sub), os.path.join(root, sub), ignore=shutil.ignore_patterns("__pycache__"))
        target = os.path.join(root, "analysis", "w3_pilot.py")
        with open(target, encoding="utf-8") as f:
            src = f.read()
        if src.count(old) != 1:
            return f"MUTATION TEXT FOUND {src.count(old)} TIMES"
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, os.path.join(root, "analysis", "test_w3_pilot.py")], capture_output=True, text=True, timeout=600)
        return "caught" if r.returncode != 0 else "NOT CAUGHT"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    bad = 0
    for name, (old, new) in MUTATIONS.items():
        res = run(old, new)
        print(f"[{res:>10}] {name}", flush=True)
        bad += res != "caught"
    print("ALL MUTATIONS CAUGHT" if not bad else f"{bad} PROBLEM(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
