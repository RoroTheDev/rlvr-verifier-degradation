"""Per-step and per-validation-round metrics from a run's per-sample reward log.

Reads <run_dir>/mixup_logs/*.jsonl (written when MIXUP_LOG_DIR is set, which the RunPod
runner always does) and reports what the proposal's week-2 logging item asks for:

    honest_pass1       fraction of rollouts passing every test (true correctness)
    honest_reward      mean honest reward (fraction of tests passed; -0.25 = no code block)
    noisy_reward       mean reward the trainer actually saw
    inverted_frac      fraction of prompt groups the noise inverted
    reward_var_noisy   mean within-group variance of the trainer's reward
    reward_var_honest  same, for the honest reward
    adv_denominator    mean within-group std (ddof=1) of the trainer's reward -- the GRPO
                       advantage denominator; a group with std 0 has zero advantage
    degenerate_noisy   fraction of groups with zero trainer-reward variance (no gradient)
    degenerate_honest  same for the honest reward (the "free" zero-variance groups that exist
                       without any noise: all-pass or all-fail)
    pass_at_k          unbiased pass@k (k = 1, 2, 4, 8, 16 where k <= rollouts per group)

A "group" is the rollouts of one prompt in one encounter, i.e. (task, encounter). Steps are
recovered by ordering groups by first-scored time and cutting every --batch-size groups,
which is exact for verl's synchronous trainer. Rows whose split is not "train" are
validation: they are split into evaluation rounds wherever scoring pauses for more than
--round-gap seconds, and reported separately (they are always scored honestly).

Optional --buckets CSV (task_id,bucket,...) adds per-bucket rows for pass@1, noise rate and
reward -- the "per-subpopulation error rate on the targeted bucket" H3 depends on.

    python analysis/run_metrics.py runs/block1 --batch-size 48 --out metrics.csv
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys

KS = (1, 2, 4, 8, 16)


def load_records(run_dir: str):
    recs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "mixup_logs", "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(at least one of k samples is correct) from c correct of n."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _var(xs):
    return statistics.pvariance(xs) if len(xs) > 1 else 0.0


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def group_records(recs):
    """[(task, encounter, [records])] ordered by the time the group was first scored."""
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["task"], r.get("encounter"))].append(r)
    return sorted(((t, e, v) for (t, e), v in groups.items()), key=lambda g: min(r["ts"] for r in g[2]))


def summarise(groups):
    """Metrics over a list of (task, encounter, records) groups."""
    calls = [r for _, _, v in groups for r in v]
    out = {
        "n_groups": len(groups),
        "n_calls": len(calls),
        "honest_pass1": sum(r["acc"] for r in calls) / len(calls),
        "honest_reward": sum(r["honest_reward"] if "honest_reward" in r else r["honest_score"] for r in calls) / len(calls),
        "noisy_reward": sum(r["score"] for r in calls) / len(calls),
        "inverted_frac": sum(1 for _, _, v in groups if any(r["mixup_flipped"] for r in v)) / len(groups),
    }
    noisy = [[r["score"] for r in v] for _, _, v in groups]
    honest = [[r.get("honest_reward", r["honest_score"]) for r in v] for _, _, v in groups]
    out["reward_var_noisy"] = statistics.fmean(_var(x) for x in noisy)
    out["reward_var_honest"] = statistics.fmean(_var(x) for x in honest)
    out["adv_denominator"] = statistics.fmean(_std(x) for x in noisy)
    out["degenerate_noisy"] = sum(1 for x in noisy if _var(x) == 0.0) / len(groups)
    out["degenerate_honest"] = sum(1 for x in honest if _var(x) == 0.0) / len(groups)
    for k in KS:
        vals = [pass_at_k(len(v), int(round(sum(r["acc"] for r in v))), k) for _, _, v in groups if len(v) >= k]
        out[f"pass_at_{k}"] = statistics.fmean(vals) if vals else ""
    return out


def split_steps(groups, batch_size: int):
    return [groups[i : i + batch_size] for i in range(0, len(groups), batch_size)]


def split_rounds(recs, gap: float):
    """Validation rows -> evaluation rounds, wherever scoring pauses for more than `gap` seconds."""
    recs = sorted(recs, key=lambda r: r["ts"])
    rounds, current, last = [], [], None
    for r in recs:
        if last is not None and r["ts"] - last > gap:
            rounds.append(current)
            current = []
        current.append(r)
        last = r["ts"]
    if current:
        rounds.append(current)
    return rounds


def load_buckets(path: str):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["task_id"])] = row.get("bucket", "")
    return out


def analyse(recs, batch_size: int = 48, round_gap: float = 120.0, buckets: dict = None):
    """Rows of {kind, step_or_round, bucket, **metrics}."""
    rows = []
    train = [r for r in recs if r.get("split", "train") == "train"]
    val = [r for r in recs if r.get("split", "train") != "train"]
    for i, step_groups in enumerate(split_steps(group_records(train), batch_size), start=1):
        rows.append({"kind": "train", "step": i, "bucket": "", **summarise(step_groups)})
        if buckets:
            by_bucket = collections.defaultdict(list)
            for t, e, v in step_groups:
                idx = v[0].get("index")
                by_bucket[buckets.get(int(idx), "(none)") if idx is not None else "(no index)"].append((t, e, v))
            for b, gs in sorted(by_bucket.items()):
                rows.append({"kind": "train", "step": i, "bucket": b, **summarise(gs)})
    for j, rnd in enumerate(split_rounds(val, round_gap), start=1):
        rows.append({"kind": "validation", "step": j, "bucket": "", **summarise(group_records(rnd))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--batch-size", type=int, default=48, help="prompt groups per training step")
    ap.add_argument("--round-gap", type=float, default=120.0, help="seconds of silence that separate validation rounds")
    ap.add_argument("--buckets", help="CSV with task_id,bucket for per-bucket rows")
    ap.add_argument("--out", help="write CSV here")
    args = ap.parse_args()

    recs = load_records(args.run_dir)
    if not recs:
        sys.exit(f"no mixup_logs found under {args.run_dir}")
    rows = analyse(recs, args.batch_size, args.round_gap, load_buckets(args.buckets) if args.buckets else None)
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out}")
    show = ("kind", "step", "bucket", "n_groups", "honest_pass1", "honest_reward", "noisy_reward", "inverted_frac",
            "adv_denominator", "degenerate_noisy", "degenerate_honest", "pass_at_1", "pass_at_16")
    print("  ".join(f"{c:>15}" for c in show))
    for r in rows:
        print("  ".join(f"{r[c]:>15.3f}" if isinstance(r.get(c), float) else f"{str(r.get(c, '')):>15}" for c in show))


if __name__ == "__main__":
    main()
