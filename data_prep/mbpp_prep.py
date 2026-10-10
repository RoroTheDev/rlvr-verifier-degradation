"""Build verl-format MBPP parquet files (train.parquet, val.parquet).

verl trains from a table with one row per prompt: the chat prompt, a data_source,
the reward model's ground truth, and free-form extra_info. verl ships a converter
for GSM8K but none for MBPP; this is that converter.

Split (Plesner et al.): the 374 MBPP "train" problems train, the 90 "validation"
problems validate.

Why train.parquet repeats the problems: the group-flip noise is redrawn every time
a prompt is *seen*. The reward function only receives a row's extra_info, not the
epoch, so each pass over the problems is written out explicitly with an
`encounter` number (0, 1, 2, ...), and verl is run with data.shuffle=False and a
single epoch. Each pass is its own random ordering of the 374 problems, chosen by
--order-seed, so training seeds see different data orders.

Row layout:
    data_source   "mbpp"
    prompt        [{"role": "user", "content": ...}]
    reward_model  {"style": "rule", "ground_truth": '{"tests": [...], "setup": ...}'}
    extra_info    {"split": "train"|"validation", "index": <MBPP task_id>, "encounter": k}

The prompt approximates the one in Plesner's reproducibility script (problem text,
the function name, one example from the first test, an instruction to answer in a
```python block); the exact wording is not published, so it is not byte-identical.

    python data_prep/mbpp_prep.py --out /tmp/mbpp --epochs 34 --order-seed 1
"""

import argparse
import json
import os
import random
import re

_FN_RE = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*(?:math\.\w+\()?\s*([A-Za-z_]\w*)\s*\(")


def parse_function_name(test: str):
    """Name of the function a MBPP assert calls, or None if it cannot be found."""
    match = _FN_RE.search(test)
    return match.group(1) if match else None


def build_prompt(text: str, tests) -> str:
    lines = [text.strip(), ""]
    name = parse_function_name(tests[0]) if tests else None
    if name:
        lines.append(f"The function must be named `{name}`.")
    if tests:
        lines.append(f"Example of the expected behaviour: `{tests[0].strip()}`")
    lines.append("Write the solution in Python 3 inside a ```python code block.")
    return "\n".join(lines)


def make_row(item: dict, split: str, encounter: int, extras: dict = None) -> dict:
    tests = list(item["test_list"])
    ground_truth = json.dumps({"tests": tests, "setup": item.get("test_setup_code") or ""})
    extra_info = {"split": split, "index": int(item["task_id"]), "encounter": int(encounter)}
    for key, value in (extras or {}).items():
        extra_info.setdefault(key, value)  # a bucket column can never overwrite split/index/encounter
    return {
        "data_source": "mbpp",
        "prompt": [{"role": "user", "content": build_prompt(item["text"], tests)}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def load_buckets(path: str) -> dict:
    """{task_id: {column: value}} from a CSV with a `task_id` column. Every other column
    becomes an extra_info field on that task's rows (numbers are parsed), so the noise can
    be targeted with MIXUP_TARGET_FIELD / _OP / _VALUE, e.g. MIXUP_TARGET_FIELD=bucket
    MIXUP_TARGET_VALUE=hard. Tasks missing from the file simply get no extra fields."""
    import csv

    def parse(v):
        try:
            return float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            return v

    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["task_id"])] = {k: parse(v) for k, v in row.items() if k != "task_id" and v != ""}
    return out


def epoch_order(task_ids, epochs: int, order_seed: int):
    """[(task_id, encounter), ...]: `epochs` back-to-back passes, each a fresh
    random permutation of all tasks, so every task appears exactly once per pass."""
    rng = random.Random(order_seed)
    order = []
    for k in range(epochs):
        ids = list(task_ids)
        rng.shuffle(ids)
        order.extend((tid, k) for tid in ids)
    return order


def build_rows(train_items, val_items, epochs: int, order_seed: int, buckets: dict = None):
    buckets = buckets or {}
    by_id = {int(it["task_id"]): it for it in train_items}
    train_rows = [
        make_row(by_id[tid], "train", k, buckets.get(tid)) for tid, k in epoch_order(sorted(by_id), epochs, order_seed)
    ]
    val_rows = [
        make_row(it, "validation", 0, buckets.get(int(it["task_id"])))
        for it in sorted(val_items, key=lambda it: int(it["task_id"]))
    ]
    return train_rows, val_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for train.parquet and val.parquet")
    ap.add_argument("--epochs", type=int, default=34, help="passes over the 374 train problems (48 prompts/step => ~7.8 steps per pass)")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--buckets", help="CSV with a task_id column; other columns are added to extra_info (for targeted noise)")
    args = ap.parse_args()

    import datasets  # imported late so the pure functions above are testable without it

    ds = datasets.load_dataset("google-research-datasets/mbpp", "full")
    buckets = load_buckets(args.buckets) if args.buckets else None
    train_rows, val_rows = build_rows(list(ds["train"]), list(ds["validation"]), args.epochs, args.order_seed, buckets)
    os.makedirs(args.out, exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(args.out, "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(args.out, "val.parquet"))
    print(f"train rows: {len(train_rows)} ({len(set(r['extra_info']['index'] for r in train_rows))} problems x {args.epochs} passes)")
    print(f"val rows:   {len(val_rows)}")


if __name__ == "__main__":
    main()
