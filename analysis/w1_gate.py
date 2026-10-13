"""Call the W1 gate from fetched run directories, by the rule fixed in docs/w1_gate_threshold.md.

Endpoint per run: V = mean validation reward at the two late evaluations (verl steps 240 and
260), in percentage points. D = mean(V_clean) - mean(V_noisy); s_p = pooled seed SD;
UB = one-sided 95% Welch upper bound on D. Outcomes (fixed before any run):

    REPLICATED                        D <= 1.0 and UB <= 2.0
    CONSISTENT BUT UNDERPOWERED       D <= 1.0 and UB >  2.0
    WITHIN SEED NOISE, ABOVE BOUND    1.0 < D <= 2 * s_p
    NOT REPLICATED                    D > 1.0 and D > 2 * s_p

Validity checks void the gate rather than fail it: every clean seed learns (late reward at
least 3 points above the step-20 evaluation), every noisy run inverted about the configured
fraction of prompt groups, no validation row was flipped, both arms are on the same reward
scale, and every validation number equals verl's own.

    python analysis/w1_gate.py --clean runs/w1-clean-s1 runs/w1-clean-s2 runs/w1-clean-s3 \\
                               --noisy runs/w1-resamp15-s1 runs/w1-resamp15-s2 runs/w1-resamp15-s3
"""

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import log_to_wandb as lw  # noqa: E402
import run_metrics as rm  # noqa: E402

MARGIN_PT = 1.0       # the paper's claim: late mean within 1 point of clean
UB_LIMIT_PT = 2.0     # how tightly 3 seeds must bound the gap before we say "replicated"
LEARN_PT = 3.0
NOISE_TOL = 0.03


# ---------- statistics (standard library only) ----------

def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    p = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t >= 0 else p


def t_quantile(q, df):
    """Student-t quantile by bisection (df may be fractional, as Welch's is)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch(clean, noisy, level=0.95):
    """Gap clean - noisy with its pooled SD and one-sided upper bound. Needs >= 2 values per arm."""
    nc, nn = len(clean), len(noisy)
    mc, mn = statistics.fmean(clean), statistics.fmean(noisy)
    vc, vn = statistics.variance(clean), statistics.variance(noisy)
    se2 = vc / nc + vn / nn
    d = mc - mn
    if se2 == 0.0:
        return {"D": d, "s_clean": 0.0, "s_noisy": 0.0, "s_p": 0.0, "se": 0.0, "df": float("inf"), "t": float("nan"), "UB": d}
    df = se2 ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vn / nn) ** 2 / (nn - 1))
    t = t_quantile(level, df)
    return {"D": d, "s_clean": math.sqrt(vc), "s_noisy": math.sqrt(vn), "s_p": math.sqrt((vc + vn) / 2.0),
            "se": math.sqrt(se2), "df": df, "t": t, "UB": d + t * math.sqrt(se2)}


def classify(d, ub, s_p):
    if d <= MARGIN_PT:
        return "REPLICATED" if ub <= UB_LIMIT_PT else "CONSISTENT BUT UNDERPOWERED"
    if d <= 2.0 * s_p:
        return "WITHIN SEED NOISE, ABOVE THE PAPER'S BOUND"
    return "NOT REPLICATED"


# ---------- one run ----------

def load_run(run_dir, late_steps):
    """-> dict with the endpoint, the per-step validation table and the validity findings of one run."""
    cfg = lw.parse_manifest(os.path.join(run_dir, "manifest.txt"))
    recs = rm.load_records(run_dir)
    if not recs:
        raise ValueError(f"{run_dir}: no per-sample records")
    rows = rm.analyse(recs, int(cfg["cfg_TRAIN_BATCH"]))
    val_rows = [r for r in rows if r["kind"] == "validation"]
    sched = lw.val_step_schedule(len(val_rows), int(cfg["cfg_STEPS"]), int(cfg["cfg_VAL_FREQ"]),
                                 cfg.get("cfg_VAL_BEFORE_TRAIN", "False") == "True")
    val = {s: r for s, r in zip(sched, val_rows)}
    missing = [s for s in late_steps if s not in val]
    if missing:
        raise ValueError(f"{run_dir}: no validation at step(s) {missing} (have {sched})")
    with open(os.path.join(run_dir, "train_log.txt"), encoding="utf-8", errors="replace") as f:
        steps, finals = lw.parse_verl_log(f.read())
    verl_lines, verl_ok = lw.verify(rows, steps, finals, cfg)
    train = [r for r in recs if r.get("split", "train") == "train"]
    vrecs = [r for r in recs if r.get("split", "train") != "train"]
    groups = rm.group_records(train)
    inverted = sum(1 for _, _, v in groups if any(r["mixup_flipped"] for r in v))
    with open(os.path.join(run_dir, "progress.txt"), encoding="utf-8", errors="replace") as f:
        url = lw.parse_wandb_url(f.read())
    first = min(val)
    return {
        "name": os.path.basename(os.path.normpath(run_dir)), "seed": cfg.get("cfg_SEED"), "mode": cfg.get("MIXUP_MODE", "clean"),
        "group_p": float(cfg.get("MIXUP_GROUP_P", 0.0)), "commit": cfg.get("team_repo_sha", "")[:10], "gpu": cfg.get("gpu", ""),
        "V": 100.0 * statistics.fmean(val[s]["honest_reward"] for s in late_steps),
        "V_pass1": 100.0 * statistics.fmean(val[s]["honest_pass1"] for s in late_steps),
        "late": {s: 100.0 * val[s]["honest_reward"] for s in late_steps},
        "first_step": first, "first": 100.0 * val[first]["honest_reward"],
        "curve": {s: 100.0 * r["honest_reward"] for s, r in val.items()},
        "inverted_frac": inverted / len(groups),
        "val_flipped": sum(1 for r in vrecs if r["mixup_flipped"]),
        "val_score_is_honest": all(r["score"] == r.get("honest_reward", r["honest_score"]) for r in vrecs),
        "scales": sorted({r.get("reward_scale") for r in recs}),
        "verl_ok": verl_ok, "verl_lines": verl_lines,
        "wandb": "https://wandb.ai/" + "/".join(url[:2]) + "/runs/" + url[2] if url else "",
    }


def validity(clean, noisy):
    """-> [(ok, text)] for the checks that void the gate."""
    out = []
    last = max(clean[0]["late"]) if clean else None
    for r in clean:
        gain = r["late"][last] - r["first"]
        out.append((gain >= LEARN_PT, f"{r['name']}: learns ({r['first']:.2f} at step {r['first_step']} -> {r['late'][last]:.2f} at step {last}, gain {gain:+.2f} pt, need >= {LEARN_PT})"))
        out.append((r["inverted_frac"] == 0.0, f"{r['name']}: clean arm has no inverted groups ({r['inverted_frac']:.4f})"))
    for r in noisy:
        out.append((abs(r["inverted_frac"] - r["group_p"]) <= NOISE_TOL, f"{r['name']}: inverted {r['inverted_frac']:.4f} of prompt groups (configured {r['group_p']}, tolerance {NOISE_TOL})"))
    for r in clean + noisy:
        out.append((r["val_flipped"] == 0 and r["val_score_is_honest"], f"{r['name']}: validation was scored honestly ({r['val_flipped']} flipped rows)"))
        out.append((r["scales"] == ["continuous"], f"{r['name']}: reward scale {r['scales']} (both arms must be continuous)"))
        out.append((r["verl_ok"], f"{r['name']}: our numbers equal verl's own"))
    seeds_c, seeds_n = sorted(r["seed"] for r in clean), sorted(r["seed"] for r in noisy)
    out.append((len(clean) >= 3 and len(noisy) >= 3, f"at least 3 seeds per arm ({len(clean)} clean, {len(noisy)} noisy)"))
    out.append((seeds_c == seeds_n and len(set(seeds_c)) == len(seeds_c), f"same distinct seeds in both arms (clean {seeds_c}, noisy {seeds_n})"))
    out.append((len({r["commit"] for r in clean + noisy}) == 1, f"one code commit for every run ({sorted({r['commit'] for r in clean + noisy})})"))
    return out


def report(clean, noisy, late_steps):
    w = welch([r["V"] for r in clean], [r["V"] for r in noisy])
    wp = welch([r["V_pass1"] for r in clean], [r["V_pass1"] for r in noisy])
    checks = validity(clean, noisy)
    valid = all(ok for ok, _ in checks)
    outcome = classify(w["D"], w["UB"], w["s_p"])
    L = []
    L.append(f"W1 gate: mean validation reward at verl steps {' and '.join(map(str, late_steps))}, percentage points")
    L.append("")
    L.append("| run | arm | seed | " + " | ".join(f"step {s}" for s in late_steps) + " | V | V (pass@1) | inverted groups | W&B |")
    L.append("|---|---|---|" + "---|" * (len(late_steps) + 4))
    for arm, runs in (("clean", clean), ("noisy", noisy)):
        for r in runs:
            L.append(f"| {r['name']} | {arm} | {r['seed']} | " + " | ".join(f"{r['late'][s]:.2f}" for s in late_steps)
                     + f" | {r['V']:.2f} | {r['V_pass1']:.2f} | {r['inverted_frac']:.3f} | {r['wandb']} |")
    L.append("")
    L.append(f"clean  mean {statistics.fmean(r['V'] for r in clean):.2f}, seed SD {w['s_clean']:.2f} (n={len(clean)})")
    L.append(f"noisy  mean {statistics.fmean(r['V'] for r in noisy):.2f}, seed SD {w['s_noisy']:.2f} (n={len(noisy)})")
    L.append(f"gap D = clean - noisy = {w['D']:+.2f} pt; pooled seed SD s_p = {w['s_p']:.2f}; SE = {w['se']:.2f}; Welch df = {w['df']:.2f}")
    L.append(f"one-sided 95% upper bound on D: {w['UB']:+.2f} pt  (paper's margin {MARGIN_PT}, bound limit {UB_LIMIT_PT}, 2 x s_p = {2 * w['s_p']:.2f})")
    L.append(f"secondary, on pass@1: D = {wp['D']:+.2f} pt, UB = {wp['UB']:+.2f} pt, s_p = {wp['s_p']:.2f}")
    L.append("")
    L.append("validity checks:")
    L += [f"  [{'ok' if ok else 'FAIL'}] {text}" for ok, text in checks]
    L.append("")
    L.append(f"OUTCOME: {outcome}" if valid else f"OUTCOME: VOID (a validity check failed; the rule would have said: {outcome})")
    return "\n".join(L), outcome, valid, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--noisy", nargs="+", required=True)
    ap.add_argument("--late-steps", nargs="+", type=int, default=[240, 260])
    ap.add_argument("--out", help="also write the report here")
    a = ap.parse_args()
    clean = [load_run(d, a.late_steps) for d in a.clean]
    noisy = [load_run(d, a.late_steps) for d in a.noisy]
    text, outcome, valid, _ = report(clean, noisy, a.late_steps)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.exit(0 if valid and outcome == "REPLICATED" else 1)


if __name__ == "__main__":
    main()
