# Bucket file: how to label tasks for targeted noise

Targeted noise needs to know which tasks belong to which subpopulation. That comes from one
CSV, committed to this repo, that the Data lane owns.

## Format

```
task_id,bucket
601,leaky
602,sound
603,leaky
```

- `task_id` is required: the MBPP task id (train 601-974, validation 511-600).
- Every other column is a label. `bucket` is the conventional name; more columns are fine
  (`task_id,bucket,difficulty`).
- One row per task. A repeated `task_id` is refused.
- A task that is not in the file, or has an empty cell, has no label and is **never** targeted,
  whatever the selector.
- A column is numeric only if every value in it is a number; otherwise the whole column is text.
  Text labels are matched exactly (`1` matches `1`, not `1.0`).

Put the file under `data_prep/buckets/` and commit it: a pod reads it from the repo at the
commit it was launched with, and the run's manifest records the file's path and hash.

`data_prep/buckets/smoke_parity.csv` (odd / even task id) is a test fixture for the plumbing,
not a real assignment.

## Launching a targeted run

Add to the usual `launch.py launch` command (`--ref` must contain the file):

```
--env BUCKETS=data_prep/buckets/<file>.csv \
--env MIXUP_TARGET_FIELD=bucket --env MIXUP_TARGET_VALUE=leaky
```

`MIXUP_TARGET_OP` is `eq` by default; `ne`, `lt`, `lte`, `gt`, `gte` also work (the ordering
ones are for numeric columns). Then choose the noise as usual:

| Noise | Settings | Persistent across epochs? |
|---|---|---|
| Whole group inverted, fixed per task | `MIXUP_MODE=group_persistent MIXUP_GROUP_P=<p>` | yes |
| Whole group inverted, redrawn each time | `MIXUP_MODE=group_resampled MIXUP_GROUP_P=<p>` | no |
| Per-sample label error, fixed per task | `MIXUP_MODE=persistent MIXUP_TPR=.. MIXUP_FPR=.. MIXUP_COVERAGE=<fraction of targeted tasks>` | yes |
| Per-sample label error, redrawn | `MIXUP_MODE=resampled MIXUP_TPR=.. MIXUP_FPR=..` | no |

A clean baseline for a group-noise arm needs `MIXUP_REWARD_SCALE=continuous` (see
`docs/w1_gate_runs.md`). The pod stops with `BUCKETS_MISSING` or `DATA_PREP_FAILED` in
`progress.txt` if the file is absent from the commit or the data builder refuses it.

## What gets logged per bucket

`python analysis/run_metrics.py <run dir> --buckets <the same csv>` adds a row per bucket to
every training step, and `analysis/log_to_wandb.py <run dir> --buckets <csv> --push` puts them in
W&B as `honest_bucket/<bucket>/...`: honest pass@1 and reward, the reward the trainer saw, the
fraction of groups inverted, and the verifier's false-positive and false-negative rates on that
bucket.

## Checked without a GPU

Real MBPP rows built with the fixture, scored by the real reward function and scorer over two
passes: with `bucket=odd` targeted at p = 0.5, 105 of 187 odd tasks were inverted and no even
task; `group_persistent` inverted the same 105 tasks in both passes while `group_resampled`
changed its set (88, then 99, 46 in common); the per-sample `persistent` mode at FPR 0.5 and
coverage 0.5 corrupted 49 of 187 odd tasks, the same 49 in both passes; no validation row was
ever touched; and with only part of the tasks labelled, the unlabelled ones were never targeted,
including under `ne`.
