# W1 gate: pass/fail threshold (written before any gate run)

**Status: PROPOSED by the Coding lane. It needs the team lead's sign-off; if they change it,
they change it in this file before the gate runs start, and the commit history shows when.**

Tracker items this answers: "Define numerically what 'within seed noise' means, before any W1 run"
and "Write the pass/fail threshold BEFORE the runs land, then call the gate".

## What is being tested

Plesner et al., *An Imperfect Verifier is Good Enough* (arXiv 2604.07666 **v2**, 27 Sep 2026):
the mean validation reward over two fixed late evaluations is within 1 percentage point of the
clean baseline for resampled group-rollout noise through 20% (v1 said 15%, "peak within 2 pts";
we follow v2, as decided by the team).

## Setup (identical in both arms except the noise)

- Task: MBPP, 374 train / 90 validation problems, `TASK=mbpp` preset in `runpod/verl_run.sh`
  (48 prompts x 16 rollouts, lr 1e-6, clip 0.20/0.28, no KL, seq-mean-token-mean loss,
  Adam wd 0.1 / beta2 0.98, 8 optimizer updates per step, 260 steps).
- Model: Qwen3-1.7B, thinking off.
- Arms: **clean** (`MIXUP_MODE=clean`, `MIXUP_REWARD_SCALE=continuous`) and **noisy**
  (`MIXUP_MODE=group_resampled`, `MIXUP_GROUP_P=0.15`).
- Seeds: 1, 2, 3 in each arm (training seed; the noise mask seed defaults to the training seed).
- Validation: every 20 steps, 16 samples per problem at T=0.7, scored honestly (never noised).

## Endpoint

For each run, **V** = the mean of the validation reward at verl steps **240 and 260** (the paper's
steps 239 and 259, 0-indexed). Validation reward = mean unit-test reward over 90 problems x 16
samples, including the -0.25 no-code-block penalty.

Let `D = mean(V_clean) - mean(V_noisy)` over the 3 seeds of each arm, `s_p` the pooled sample SD of V
across seeds (`sqrt((s_clean^2 + s_noisy^2) / 2)`), and `UB` the one-sided 95% upper confidence bound
on `D` (Welch). All in percentage points of reward.

## Outcomes, fixed in advance

1. **REPLICATED**: `D <= 1.0` and `UB <= 2.0`.
2. **CONSISTENT BUT UNDERPOWERED**: `D <= 1.0` but `UB > 2.0`. We do **not** claim replication; the
   result says the gap is small but 3 seeds cannot bound it. Add seeds, or reduce seed noise, before moving on.
3. **WITHIN SEED NOISE, ABOVE THE PAPER'S BOUND**: `1.0 < D <= 2 * s_p`. Not replicated at the paper's
   tolerance, but indistinguishable from seed variation. Report as such; proceed only with the team lead's agreement.
4. **NOT REPLICATED**: `D > 2 * s_p` and `D > 1.0`. **Stop and debug** (tracker gate: "else STOP and debug.
   No progress on a near-miss").

## Validity checks (a failure voids the gate rather than failing it)

- **It learns:** in every clean seed, V at step 260 exceeds the validation reward at step 20 by at least 3 points.
- **The noise happened:** in every noisy run, the fraction of inverted groups, measured from the per-sample
  log, is within 0.15 +/- 0.03 over the run.
- **Honest validation:** no validation row was flipped, and every noisy run's validation scores equal the
  honest reward.
- **Cost logged:** dollars and wall-clock per run are recorded for the W4 budget.

## Deviations from the paper, to state wherever the result is quoted

Qwen3-**1.7B** (paper: 8B primary; 4B-9B across families); thinking off; verl instead of their
SLIME-based framework; 6-prompt mini-batches (8 updates per step) inferred from "global batch size 96", which the
paper does not state; MBPP prompt wording approximated; evaluation response length equals the 4096 training
cap (paper: 8192 at evaluation); 3 seeds per arm (paper: 4 for the Qwen3-8B baseline, 3-6 for noise settings).
Only one noise rate (15%) is tested at the gate.
