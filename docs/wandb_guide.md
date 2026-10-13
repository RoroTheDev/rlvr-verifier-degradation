# Reading the runs in W&B

Project: <https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation>

## Which runs are which

| Group | Runs | Use |
|---|---|---|
| `w1-gate` | `w1-clean-s1..3`, `w1-resamp15-s1..3` | The Week 1 gate: 3 clean seeds and 3 seeds with 15% resampled group noise. Qwen3-1.7B, thinking off, MBPP, 260 steps. |
| `preset-check`, `block1-smoke`, none | 4 runs of 1 to 4 steps | Plumbing tests from before the gate. Not data. |

Every gate run carries the tags `w1-gate`, `qwen3-1.7b`, `nothink`, `mbpp`. The run name is the
run directory name everywhere else (launcher, fetched files, `docs/w1_gate_threshold.md`).

## Which numbers are honest

The trainer's own reward curve is **not** accuracy in a noisy run: it is the reward after the
noise was applied.

| Key | What it is | Honest? |
|---|---|---|
| `critic/score/mean` | mean reward the trainer learned from, per step | **No** in a noisy run (equal to honest in a clean run) |
| `val-aux/mbpp/reward/mean@16` | validation reward: mean fraction of tests passed over 90 problems x 16 samples, -0.25 for no code block. **The gate's endpoint.** | Yes (validation is never noised) |
| `val-core/mbpp/acc/mean@16` | validation pass@1 (all tests pass) | Yes |
| `honest/honest_reward`, `honest/honest_pass1` | training reward and pass@1 before noise, per step | Yes |
| `honest/noisy_reward` | the same number as `critic/score/mean`, recomputed from the per-sample log | No |
| `honest/inverted_frac` | fraction of that step's prompt groups whose rewards were inverted | - |
| `honest/false_pos_rate`, `honest/false_neg_rate` | the verifier's error rates as the trainer saw them: really-wrong samples rewarded as fully correct, and really-correct samples not rewarded as correct (runs pushed from 2026-10-13 on) | - |
| `honest/flip_rate`, `honest/eligible_frac` | fraction of samples the noise changed, and fraction on tasks the targeting selector picks | - |
| `honest_bucket/<bucket>/*` | the same metrics per bucket, when a bucket file is used (`docs/bucket_file.md`) | as above |
| `honest/pass_at_1 .. pass_at_16` | unbiased pass@k over the 16 rollouts of each prompt | Yes |
| `honest/adv_denominator` | mean within-group std of the trainer's reward (the GRPO advantage denominator) | - |
| `honest/reward_var_noisy`, `honest/reward_var_honest` | mean within-group reward variance, as trained on / before noise | - |
| `honest/degenerate_noisy`, `honest/degenerate_honest` | fraction of groups with zero reward variance (no gradient) | - |
| `honest_val/*` | the same set per validation round | Yes |
| `actor/entropy`, `actor/pg_loss`, `actor/grad_norm`, ... | verl's own training diagnostics | - |

## How to plot

- Honest training curves: x-axis `honest/step`. Honest validation curves: x-axis `honest_val/step`
  (the verl global step of that evaluation: 20, 40, ..., 260). Everything verl logs uses W&B's
  default step.
- The `honest/*` and `honest_val/*` keys appear when a run has **finished**: they are computed
  from the run's per-sample log and pushed by `analysis/log_to_wandb.py`, which first checks our
  per-step noisy reward and per-evaluation pass@1 and reward against verl's own and refuses to
  push on a mismatch. During a run only verl's keys are live.

## Provenance

Each run's config has `manifest/*` keys: the repo commit (`manifest/team_repo_sha`), the verl
commit, hashes of the reward function, the scorer and both data files, the GPU, package
versions, and the effective value of every knob (`manifest/cfg_*`, `manifest/MIXUP_*`).
A number is reportable only if its run has these.

## The gate result

`python analysis/w1_gate.py --clean <3 run dirs> --noisy <3 run dirs>` applies the rule in
`docs/w1_gate_threshold.md` and prints the per-run endpoints, the seed SD, the gap, its
one-sided 95% bound, and the validity checks.
