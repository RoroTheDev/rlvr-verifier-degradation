"""Apply the W3 pilot rule in docs/w3_pilot_threshold.md to fetched run directories.

Persistent (P) against resampled (R) group noise at the same rate. Endpoint per run: V = mean
honest validation reward at verl steps 240 and 260, in points. G = mean(V_R) - mean(V_P)
(positive: persistent noise did more harm); s_p = pooled seed SD of the two arms;
T = max(2 * s_p, 1.0). Outcomes, fixed before any persistent run:

    H2 SIGNAL                     G >= T
    NO LARGE PERSISTENCE EFFECT   -T < G < T
    REVERSED                      G <= -T

Validity checks void the pilot rather than decide it: matched implementation (manifest hashes,
library versions, every setting except the noise mode), matched rate, persistence in P and none
in R, honest validation, one reward scale, numbers equal verl's, complete single-boot runs, and
the same seeds in both arms. The Data analysis lane calls the pilot; this script only applies
the rule.

    python analysis/w3_pilot.py --resampled RUNS/w1-resamp15-s1 RUNS/w1-resamp15-s2 RUNS/w1-resamp15-s3 \\
                                --persistent RUNS/w3-persist15-s1 RUNS/w3-persist15-s2 RUNS/w3-persist15-s3 \\
                                [--clean RUNS/w1-clean-s1 ...] [--state BATCH.state.json ...]
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import log_to_wandb as lw  # noqa: E402
import run_metrics as rm  # noqa: E402
import w1_gate as g  # noqa: E402

FLOOR_PT = 1.0               # T never drops below the paper's 1-point margin
NOISE_TOL = 0.03             # inverted share of groups must be within p +/- this (as at W1)
RESAMPLED_CHANGING_MIN = 0.90
SAME_KEYS = ("runner_script_sha256", "team_repo_sha", "reward_file_sha256", "mbpp_scoring_sha256", "val_parquet_sha256",
             "verl_sha", "verl_uv_lock_sha256", "python", "torch", "vllm", "ray", "transformers", "flash-attn",
             "tensordict", "numpy", "datasets")
# Not compared across runs: the seed (compared per seed instead) and the two timers that only
# decide when an idle pod stops itself.
IGNORED_CFG = ("cfg_SEED", "cfg_IDLE_STOP_MIN", "cfg_MAX_RUNTIME_MIN")
IGNORED_MIXUP = ("MIXUP_MODE", "MIXUP_LOG_DIR", "MIXUP_MASK_SEED")   # mask seed is checked against each run's seed

H2, NONE, REVERSED = "H2 SIGNAL", "NO LARGE PERSISTENCE EFFECT", "REVERSED"


# ---------- the persistent set, predicted independently of the reward function ----------

def unit_interval(mask_seed, salt, key):
    """Same formula as _deterministic_unit_interval in reward_functions/persistence_reward.py."""
    digest = hashlib.sha256(f"{mask_seed}|{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def task_key(data_source, split, index):
    """Same string as _task_identity for a row whose extra_info has an index."""
    return json.dumps({"data_source": data_source, "anchor": ["index", split, index]}, sort_keys=True, default=str)


def predicted_set(indices, data_source, mask_seed, p, split="train"):
    return {i for i in indices if unit_interval(mask_seed, "group_flip", task_key(data_source, split, i)) < p}


# ---------- the rule ----------

def threshold(s_p):
    return max(2.0 * s_p, FLOOR_PT)


def classify(gap, s_p):
    t = threshold(s_p)
    if gap >= t:
        return H2
    if gap <= -t:
        return REVERSED
    return NONE


# ---------- one run ----------

def load_run(run_dir, late_steps):
    """w1_gate.load_run plus the manifest, per-task inversion status, completeness and the mixed-group share."""
    r = g.load_run(run_dir, late_steps)
    cfg = lw.parse_manifest(os.path.join(run_dir, "manifest.txt"))
    train = [x for x in rm.load_records(run_dir) if x.get("split", "train") == "train"]
    status = collections.defaultdict(set)   # task index -> inverted? over the passes where it could be inverted
    sources = set()
    mixed = mixed_inv = 0
    for _, _, v in rm.group_records(train):
        idx = v[0]["index"]
        sources.add(v[0].get("data_source"))
        inv = any(x["mixup_flipped"] for x in v)
        if any(x["honest_score"] >= 0 for x in v):   # an all-format-failure group can never be inverted
            status[idx].add(inv)
        if len({x["honest_score"] for x in v}) > 1:  # only these carry a GRPO gradient
            mixed += 1
            mixed_inv += inv
    inverted_tasks = {i for i, s in status.items() if True in s}
    pred = None
    if len(sources) == 1 and cfg.get("MIXUP_MASK_SEED") is not None and cfg.get("MIXUP_GROUP_P"):
        pred = predicted_set(status, next(iter(sources)), cfg["MIXUP_MASK_SEED"], float(cfg["MIXUP_GROUP_P"]))
    with open(os.path.join(run_dir, "progress.txt"), encoding="utf-8", errors="replace") as f:
        prog = f.read()
    done = re.findall(r"STAGE:DONE(?: exit=(-?\d+))?", prog)
    with open(os.path.join(run_dir, "train_log.txt"), encoding="utf-8", errors="replace") as f:
        steps, _ = lw.parse_verl_log(f.read())
    total = int(cfg.get("cfg_STEPS", 0))
    r.update({
        "cfg": cfg,
        "n_tasks": len(status),
        "constant_frac": sum(1 for s in status.values() if len(s) == 1) / len(status) if status else 0.0,
        "changing_frac": sum(1 for s in status.values() if len(s) == 2) / len(status) if status else 0.0,
        "inverted_tasks": inverted_tasks,
        "predicted_tasks": pred,
        "mixed_inverted_share": mixed_inv / mixed if mixed else float("nan"),
        "boots": len(re.findall(r"STAGE:BOOT", prog)),
        "exit": int(done[-1]) if done and done[-1] != "" else None,
        "all_steps": total > 0 and sorted(steps) == list(range(1, total + 1)),
        "total_steps": total,
    })
    return r


# ---------- validity ----------

def _show(vals):
    return ", ".join(sorted(str(v)[:12] for v in vals))


def _gpu(r):
    return r["cfg"].get("gpu", "").split(",")[0].strip()


def validity(res, per, accept_runner_diff=False, accept_gpu_diff=False):
    """-> [(ok, text)] for the checks that void the pilot."""
    out = []
    runs = res + per
    # 1. matched implementation
    for k in SAME_KEYS:
        vals = {r["cfg"].get(k) for r in runs}
        if k == "runner_script_sha256" and len(vals) > 1 and accept_runner_diff:
            out.append((True, f"{k} DIFFERS ({', '.join(sorted(map(str, vals)))}); accepted with --accept-runner-diff, quote both hashes"))
            continue
        out.append((len(vals) == 1, f"same {k} in every run ({_show(vals)})"))
    gpus = {_gpu(r) for r in runs}
    arm_gpus = [{_gpu(r) for r in arm} for arm in (res, per)]
    if len(gpus) > 1 and accept_gpu_diff:
        same_within = all(len(s) == 1 for s in arm_gpus)
        out.append((same_within, f"GPU model DIFFERS between arms (resampled {_show(arm_gpus[0])}, persistent {_show(arm_gpus[1])}); "
                                 "accepted with --accept-gpu-diff, quote both" + ("" if same_within else " -- but an arm mixes GPU models")))
    else:
        out.append((len(gpus) == 1, f"same GPU model in every run ({_show(gpus)})"))
    for prefix, ignored in (("cfg_", IGNORED_CFG), ("MIXUP_", IGNORED_MIXUP)):
        keys = sorted({k for r in runs for k in r["cfg"] if k.startswith(prefix) and k not in ignored})
        diff = [k for k in keys if len({r["cfg"].get(k) for r in runs}) > 1]
        out.append((not diff, f"every {prefix}* setting equal across runs except {', '.join(ignored)}"
                               + (f" -- differ: {diff}" if diff else f" ({len(keys)} settings)")))
    for r in res:
        out.append((r["mode"] == "group_resampled", f"{r['name']}: mode {r['mode']} (resampled arm needs group_resampled)"))
    for r in per:
        out.append((r["mode"] == "group_persistent", f"{r['name']}: mode {r['mode']} (persistent arm needs group_persistent)"))
    for r in runs:
        out.append((r["cfg"].get("MIXUP_MASK_SEED") == r["seed"], f"{r['name']}: mask seed {r['cfg'].get('MIXUP_MASK_SEED')} equals training seed {r['seed']}"))
    by_seed = collections.defaultdict(set)
    for r in runs:
        by_seed[r["seed"]].add(r["cfg"].get("train_parquet_sha256"))
    for seed, vals in sorted(by_seed.items(), key=lambda kv: str(kv[0])):
        out.append((len(vals) == 1, f"seed {seed}: same train parquet (task order) in both arms ({_show(vals)})"))
    # 2. matched rate
    for r in runs:
        out.append((abs(r["inverted_frac"] - r["group_p"]) <= NOISE_TOL,
                    f"{r['name']}: inverted {r['inverted_frac']:.4f} of training groups (configured {r['group_p']}, tolerance {NOISE_TOL})"))
    # 3. persistence in P, none in R
    for r in per:
        out.append((r["constant_frac"] == 1.0, f"{r['name']}: every task has one status on every pass ({r['constant_frac']:.4f} of {r['n_tasks']} tasks)"))
        pred = r["predicted_tasks"]
        same = pred is not None and r["inverted_tasks"] == pred
        detail = "no prediction (mask seed, rate or data source missing)" if pred is None else \
            f"{len(r['inverted_tasks'])} inverted, {len(pred)} predicted, {len(r['inverted_tasks'] ^ pred)} differ"
        out.append((same, f"{r['name']}: inverted tasks equal the set the mask seed predicts ({detail})"))
    for r in res:
        out.append((r["changing_frac"] >= RESAMPLED_CHANGING_MIN,
                    f"{r['name']}: {r['changing_frac']:.4f} of tasks change status across passes (need >= {RESAMPLED_CHANGING_MIN})"))
    # 4-7. honest validation, one scale, verl agrees, complete single-boot runs
    for r in runs:
        out.append((r["val_flipped"] == 0 and r["val_score_is_honest"], f"{r['name']}: validation was scored honestly ({r['val_flipped']} flipped rows)"))
        out.append((r["scales"] == ["continuous"], f"{r['name']}: reward scale {r['scales']} (every run must be continuous)"))
        out.append((r["verl_ok"], f"{r['name']}: our numbers equal verl's own"))
        out.append((r["exit"] == 0 and r["boots"] == 1 and r["all_steps"],
                    f"{r['name']}: complete single-boot run (exit {r['exit']}, {r['boots']} boot(s), every step 1..{r['total_steps']}: {r['all_steps']})"))
    # 8. seeds
    sr, sp = sorted(r["seed"] for r in res), sorted(r["seed"] for r in per)
    out.append((len(res) >= 3 and len(per) >= 3, f"at least 3 seeds per arm ({len(res)} resampled, {len(per)} persistent)"))
    out.append((sr == sp and len(set(sr)) == len(sr), f"same distinct seeds in both arms (resampled {sr}, persistent {sp})"))
    return out


# ---------- report ----------

def costs(state_paths):
    """{run name: (hours, dollars, data centre)} from fleet.py batch state files."""
    out = {}
    for p in state_paths or ():
        with open(p, encoding="utf-8") as f:
            for name, s in json.load(f).get("runs", {}).items():
                if all(s.get(k) is not None for k in ("launched", "finished", "cost")):
                    out[name] = ((s["finished"] - s["launched"]) / 3600.0, s["cost"], s.get("dc", ""))
    return out


def report(res, per, late_steps, clean=None, accept_runner_diff=False, cost=None, accept_gpu_diff=False):
    w = g.welch([r["V"] for r in res], [r["V"] for r in per])
    wp = g.welch([r["V_pass1"] for r in res], [r["V_pass1"] for r in per])
    lb = w["D"] - (w["t"] * w["se"] if w["se"] > 0 else 0.0)
    checks = validity(res, per, accept_runner_diff, accept_gpu_diff)
    valid = all(ok for ok, _ in checks)
    outcome = classify(w["D"], w["s_p"])
    cost = cost or {}
    L = [f"W3 pilot: persistent vs resampled group noise, mean validation reward at verl steps {' and '.join(map(str, late_steps))}, points", ""]
    L.append("| run | arm | seed | " + " | ".join(f"step {s}" for s in late_steps)
             + " | V | V (pass@1) | inverted groups | inverted share of mixed groups | tasks inverted | hours | $ | W&B |")
    L.append("|---|---|---|" + "---|" * (len(late_steps) + 8))
    for arm, runs in (("clean", clean or []), ("resampled", res), ("persistent", per)):
        for r in runs:
            h, d, _ = cost.get(r["name"], (None, None, None))
            L.append(f"| {r['name']} | {arm} | {r['seed']} | " + " | ".join(f"{r['late'][s]:.2f}" for s in late_steps)
                     + f" | {r['V']:.2f} | {r['V_pass1']:.2f} | {r['inverted_frac']:.4f} | {r['mixed_inverted_share']:.4f}"
                     + f" | {len(r['inverted_tasks'])}/{r['n_tasks']} | {'' if h is None else f'{h:.2f}'} | {'' if d is None else f'{d:.2f}'} | {r['wandb']} |")
    L.append("")
    L.append(f"resampled  mean {statistics.fmean(r['V'] for r in res):.2f}, seed SD {w['s_clean']:.2f} (n={len(res)})")
    L.append(f"persistent mean {statistics.fmean(r['V'] for r in per):.2f}, seed SD {w['s_noisy']:.2f} (n={len(per)})")
    L.append(f"gap G = resampled - persistent = {w['D']:+.2f} pt (positive: persistent did more harm); pooled seed SD s_p = {w['s_p']:.2f}")
    L.append(f"threshold T = max(2 x s_p, {FLOOR_PT}) = {threshold(w['s_p']):.2f} pt")
    L.append(f"secondary: one-sided 95% Welch lower bound on G = {lb:+.2f} pt (SE {w['se']:.2f}, df {w['df']:.2f})")
    L.append(f"secondary: on pass@1, G = {wp['D']:+.2f} pt, s_p = {wp['s_p']:.2f}, rule would say {classify(wp['D'], wp['s_p'])}")
    seeds = sorted({r["seed"] for r in res} & {r["seed"] for r in per}, key=str)
    if seeds:
        vr, vp = {r["seed"]: r["V"] for r in res}, {r["seed"]: r["V"] for r in per}
        L.append("secondary: per-seed R - P (same task order, not paired replicates): " + ", ".join(f"seed {s} {vr[s] - vp[s]:+.2f}" for s in seeds))
    if clean:
        wc = g.welch([r["V"] for r in clean], [r["V"] for r in per])
        s3 = math.sqrt((wc["s_clean"] ** 2 + w["s_clean"] ** 2 + w["s_noisy"] ** 2) / 3.0)
        o3 = classify(w["D"], s3)
        L.append(f"secondary: clean - persistent = {wc['D']:+.2f} pt (W1 clean arm; its 'it learns' check is still open)")
        L.append(f"secondary: SD pooled over clean, R and P = {s3:.2f}, T = {threshold(s3):.2f}, rule would say {o3}"
                 + ("" if o3 == outcome else "  <-- DIFFERS from the primary outcome; report both"))
    L.append("")
    L.append("validity checks:")
    L += [f"  [{'ok' if ok else 'FAIL'}] {text}" for ok, text in checks]
    L.append("")
    L.append(f"OUTCOME: {outcome}" if valid else f"OUTCOME: VOID (a validity check failed; the rule would have said: {outcome})")
    return "\n".join(L), outcome, valid, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resampled", nargs="+", required=True)
    ap.add_argument("--persistent", nargs="+", required=True)
    ap.add_argument("--clean", nargs="*", default=[], help="W1 clean runs, for the secondary comparisons only")
    ap.add_argument("--state", nargs="*", default=[], help="fleet.py batch state files, for hours and dollars per run")
    ap.add_argument("--late-steps", nargs="+", type=int, default=[240, 260])
    ap.add_argument("--accept-runner-diff", action="store_true",
                    help="only if the Data analysis lane accepted a runner-hash difference in docs/w3_pilot_threshold.md")
    ap.add_argument("--accept-gpu-diff", action="store_true",
                    help="the arms ran on different GPU models, as recorded before launch in docs/w3_pilot_threshold.md")
    ap.add_argument("--out", help="also write the report here")
    a = ap.parse_args()
    res = [load_run(d, a.late_steps) for d in a.resampled]
    per = [load_run(d, a.late_steps) for d in a.persistent]
    clean = [load_run(d, a.late_steps) for d in a.clean]
    text, _, valid, _ = report(res, per, a.late_steps, clean, a.accept_runner_diff, costs(a.state), a.accept_gpu_diff)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
