"""Push honest per-step / per-validation-round metrics (analysis/run_metrics.py) into a run's W&B run.

W&B's own critic/score/mean is the NOISY reward the trainer saw. This adds the honest
numbers next to it, on their own x-axes, and the run manifest as config:

    honest/*            per training step   (x-axis honest/step)
    honest_val/*        per validation round (x-axis honest_val/step = verl global step)
    honest_bucket/<b>/* per-bucket rows when --buckets is given

The target run is read from <run_dir>/progress.txt (WANDB_URL:...), so it works on a
fetched run directory with no other arguments. Nothing is logged unless --push is given;
the default is to check our numbers against verl's own (critic/score/mean per step,
val-core/*/acc/mean@N per validation) and print the result.

    python analysis/log_to_wandb.py runs/NAME                 # verify only
    python analysis/log_to_wandb.py runs/NAME --push          # verify, then log (WANDB_API_KEY set)
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics as rm  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
WANDB_URL = re.compile(r"https://wandb\.ai/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/runs/([A-Za-z0-9]+)")


def parse_manifest(path):
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                cfg[k.strip()] = v
    return cfg


def cfg_int(cfg, key):
    return int(cfg[key]) if key in cfg else None


def parse_wandb_url(progress_text):
    m = WANDB_URL.search(progress_text)
    return m.groups() if m else None


NP_SCALAR = re.compile(r"^np\.\w+\((.*)\)$")


def _num(s):
    """A logged value as a float. verl prints mid-run validation values as np.float64(0.5), not 0.5."""
    s = s.strip()
    m = NP_SCALAR.match(s)
    try:
        return float(m.group(1) if m else s)
    except ValueError:
        return None


def parse_verl_log(text):
    """-> ({step: {metric: value}}, [(step_or_None, {val metric: value})]).

    Step lines look like `step:N - a:1 - b:2`. Validation metrics appear either inside a step
    line or in a "Final validation metrics: {...}" dict, which the console wraps over several
    pretty-printed chunks and interleaves with progress bars.
    """
    steps, finals = {}, []
    chunks = []
    for line in ANSI.sub("", text).splitlines():
        line = re.sub(r"^\(\w+ pid=\d+\) ", "", line)
        m = re.match(r"step:(\d+) - (.*)$", line)
        if m:
            d = {}
            for part in m.group(2).split(" - "):
                k, _, v = part.partition(":")
                if _num(v) is not None:
                    d[k.strip()] = _num(v)
            steps[int(m.group(1))] = d
            continue
        if "Training Progress" in line or "pid=" in line:
            continue
        s = line.strip()
        if s.startswith(('("', '"')) or chunks:
            chunks.append(re.sub(r'^\(?\s*"|"\)?\s*$', "", s) if s else "")
    joined = "".join(chunks)
    for block in re.findall(r"Final validation metrics: \{(.*?)\}", joined):
        vals = {k: float(v) for k, v in re.findall(r"'(val-[^']+)':\s*np\.float64\(([-0-9.e]+)\)", block)}
        finals.append((None, vals))
    return steps, finals


def val_step_schedule(n_rounds, steps_total, val_freq, before_train):
    if steps_total is None or val_freq is None:
        raise ValueError("manifest has no cfg_STEPS/cfg_VAL_FREQ (older run): cannot place validation rounds")
    sched = [0] if before_train else []
    k = 1
    while val_freq > 0 and k * val_freq <= steps_total:
        sched.append(k * val_freq)
        k += 1
    if sched[-1:] != [steps_total]:
        sched.append(steps_total)
    if len(sched) != n_rounds:
        raise ValueError(f"expected {len(sched)} validation rounds ({sched}) but the log has {n_rounds}")
    return sched


def build_payloads(rows, cfg, n_val_steps=None):
    """-> (train_payloads, val_payloads); each a flat dict of W&B keys incl. its x-axis."""
    train, val = [], []
    skip = {"kind", "step", "bucket", "n_groups", "n_calls"}
    val_rows = [r for r in rows if r["kind"] == "validation"]
    sched = None
    if val_rows:
        sched = val_step_schedule(len(val_rows), cfg_int(cfg, "cfg_STEPS"), cfg_int(cfg, "cfg_VAL_FREQ"),
                                  cfg.get("cfg_VAL_BEFORE_TRAIN", "False") == "True")
    by_step = {}
    for r in rows:
        vals = {k: v for k, v in r.items() if k not in skip and isinstance(v, (int, float))}
        if r["kind"] == "train":
            p = by_step.setdefault(r["step"], {"honest/step": r["step"]})
            if r["bucket"]:
                p.update({f"honest_bucket/{r['bucket']}/{k}": v for k, v in vals.items()})
                p[f"honest_bucket/{r['bucket']}/n_groups"] = r["n_groups"]
            else:
                p.update({f"honest/{k}": v for k, v in vals.items()})
                p["honest/n_groups"] = r["n_groups"]
        else:
            p = {f"honest_val/{k}": v for k, v in vals.items()}
            p["honest_val/step"] = sched[r["step"] - 1]
            val.append(p)
    train = [by_step[s] for s in sorted(by_step)]
    return train, val


def verify(rows, verl_steps, verl_finals, cfg, tol=1e-6):
    """Compare our numbers with verl's. Returns (report lines, ok)."""
    lines, ok = [], True
    train = [r for r in rows if r["kind"] == "train" and not r["bucket"]]
    diffs = []
    for r in train:
        v = verl_steps.get(r["step"], {}).get("critic/score/mean")
        if v is not None:
            diffs.append(abs(r["noisy_reward"] - v))
    if diffs:
        worst = max(diffs)
        lines.append(f"train: noisy_reward vs critic/score/mean over {len(diffs)}/{len(train)} steps, max |diff| = {worst:.2e}")
        ok &= worst <= tol
    else:
        lines.append("train: no critic/score/mean in the verl log to compare against")
        ok = False
    val_rows = [r for r in rows if r["kind"] == "validation"]
    if val_rows:
        try:
            sched = val_step_schedule(len(val_rows), cfg_int(cfg, "cfg_STEPS"), cfg_int(cfg, "cfg_VAL_FREQ"),
                                      cfg.get("cfg_VAL_BEFORE_TRAIN", "False") == "True")
        except ValueError as e:
            return lines + [f"val: NOT verified ({e})"], False
        n = cfg.get("cfg_VAL_N", "1")
        for r, s in zip(val_rows, sched):
            src = verl_steps.get(s, {}) | (verl_finals[-1][1] if s == sched[-1] and verl_finals else {})
            # pass@1 must be there; the reward (the W1 gate's endpoint) is checked whenever verl reports it
            for ours, suffix, required in (("honest_pass1", f"/acc/mean@{n}", True), ("honest_reward", f"/reward/mean@{n}", False)):
                keys = [m for m in src if m.endswith(suffix)]
                if not keys:
                    if required:
                        lines.append(f"val step {s}: no {suffix[1:]} in the verl log")
                        ok = False
                    continue
                d = abs(r[ours] - src[keys[0]])
                lines.append(f"val step {s}: {ours} {r[ours]:.6f} vs {keys[0]} {src[keys[0]]:.6f} (diff {d:.2e})")
                ok &= d <= tol
    return lines, ok


def push(run_dir, rows, cfg, url, wandb_module=None):
    """Log the honest metrics into the existing W&B run `url` and set the manifest as config."""
    wb = wandb_module
    if wb is None:
        import wandb as wb  # lazy: only needed to push
    entity, project, run_id = url
    train, val = build_payloads(rows, cfg)
    run = wb.init(entity=entity, project=project, id=run_id, resume="allow")
    run.define_metric("honest/step")
    run.define_metric("honest/*", step_metric="honest/step")
    run.define_metric("honest_bucket/*", step_metric="honest/step")
    run.define_metric("honest_val/step")
    run.define_metric("honest_val/*", step_metric="honest_val/step")
    run.config.update({f"manifest/{k}": v for k, v in cfg.items()}, allow_val_change=True)
    for p in train:
        run.log(p)
    for p in val:
        run.log(p)
    run.finish()
    return len(train), len(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--buckets", help="CSV with task_id,bucket for per-bucket metrics")
    ap.add_argument("--push", action="store_true", help="log to W&B (default: verify only)")
    ap.add_argument("--force", action="store_true", help="push even if the check against verl fails")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, help="prompt groups per step (default: cfg_TRAIN_BATCH from the manifest)")
    a = ap.parse_args()

    cfg = parse_manifest(os.path.join(a.run_dir, "manifest.txt"))
    recs = rm.load_records(a.run_dir)
    if not recs:
        sys.exit(f"no mixup_logs under {a.run_dir}")
    if not a.batch_size and "cfg_TRAIN_BATCH" not in cfg:
        sys.exit("manifest has no cfg_TRAIN_BATCH (older run); pass --batch-size")
    rows = rm.analyse(recs, a.batch_size or int(cfg["cfg_TRAIN_BATCH"]), buckets=rm.load_buckets(a.buckets) if a.buckets else None)
    with open(os.path.join(a.run_dir, "train_log.txt"), encoding="utf-8", errors="replace") as f:
        steps, finals = parse_verl_log(f.read())
    lines, ok = verify(rows, steps, finals, cfg, a.tol)
    print("\n".join(lines))
    print("CHECK", "PASSED" if ok else "FAILED")
    if not a.push:
        return
    if not ok and not a.force:
        sys.exit("not pushing: our numbers disagree with verl's (use --force to override)")
    with open(os.path.join(a.run_dir, "progress.txt"), encoding="utf-8") as f:
        url = parse_wandb_url(f.read())
    if not url:
        sys.exit("no WANDB_URL in progress.txt; was the run launched with W&B on?")
    nt, nv = push(a.run_dir, rows, cfg, url)
    print(f"logged {nt} train steps and {nv} validation rounds to {'/'.join(url)}")


if __name__ == "__main__":
    main()
