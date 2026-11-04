# W3 pilot: pass/fail rule (written before any pilot run)

**Status: PROPOSED by the Coding lane, pending sign-off. The Data analysis lane calls the pilot, not
the Coding lane. Changes go in this file; if a change lands after the first pilot run has started,
record the date here and have the report quote both versions of the rule.**

Tracker items this answers: W3 "Pilot: 1 family x 2 regimes x 3 seeds, matched rate" and its gate,
"persistent-vs-resampled gap under 2x the seed std: stop pushing H2, pivot the headline to H3/H5".

## Deviation recorded before launch (2026-11-03, before any persistent run started)

**The persistent arm runs on an NVIDIA A40, not an L40S.** At launch time RunPod had no secure-cloud
L40S anywhere: six attempts over 15 minutes were refused, and its stock query showed none. Rather than
wait, the Coding lane switched the persistent arm to the cheapest secure-cloud GPU in stock, an A40 at
$0.59/hr. A100 was the other option, at about $28 for three runs, over the pilot budget. All three
persistent seeds run on A40. The resampled arm ran on L40S.

- Everything else stays as above: commit, runner hash, verl, libraries, every training setting.
  `MAX_RUNTIME_MIN` is raised from 720 to 960 because an A40 run takes about 9 hours. That timer only
  decides when a pod stops itself.
- Both GPUs have 48 GB and train in bf16. Run-to-run numerical noise is already in the seed SD,
  because vLLM sampling differs on every pod. A systematic GPU effect on the endpoint cannot be ruled
  out, though, so **the GPU difference must be stated wherever the result is quoted.**
- Check 1's GPU-model condition is applied with `--accept-gpu-diff`. Each arm must still use a single
  GPU model.
- If the outcome is within about 1 point of `T` either way, one resampled seed re-run on A40 (about
  $5) would show whether the GPU moved the endpoint. The Data analysis lane decides whether to buy it.

## What is being tested

H2: at the same error rate, **persistent** verifier error (the same prompts are always mis-scored)
harms GRPO more than **resampled** error (a fresh coin every time a prompt is seen).

## Setup

- **Resampled arm (R):** the three W1 gate runs `w1-resamp15-s1/s2/s3` (`MIXUP_MODE=group_resampled`,
  `MIXUP_GROUP_P=0.15`), finished 2026-10-13. Not re-run; numbers in `docs/w1_gate_runs.md`.
- **Persistent arm (P):** three new runs `w3-persist15-s1/s2/s3` (`MIXUP_MODE=group_persistent`,
  `MIXUP_GROUP_P=0.15`, seeds 1, 2, 3).
- **Everything else is identical to R:** team repo commit `ca336b3` on the pod; the same runner script,
  launched from a worktree of tag `w1-gate-runs` (`runner_script_sha256` `3951eddc...`, as in all six
  gate manifests); verl `8718ca3`; Qwen3-1.7B, thinking off; `TASK=mbpp` preset (48 prompts x 16
  rollouts, 260 steps, validation every 20 steps at 16 samples, T=0.7); continuous reward;
  `DYN_BSZ=True`, `MAX_TOKENS_PER_GPU=8192`; one L40S per run. The noise mask seed defaults to the
  training seed.
- **The only difference between the arms:** P keys the group coin on the task alone, so a task that is
  inverted is inverted on every pass; R redraws it on every pass. Same structure (whole-group inversion,
  format failures never inverted), same marginal rate.
- The task order depends only on the seed (R seed k and the W1 clean seed k have the same train parquet
  hash), so P seed k sees exactly the same sequence of prompts as R seed k.

## Known before any pilot run

The persistent set is fixed by hashing (mask seed, task), so it was computed in advance from the gate's
own reward code (`ca336b3`). The task keys were checked against the task hashes logged by the R runs
(374 of 374 match for each seed).

| Seed | Tasks in P's set | Share of 374 | Expected inverted groups, P | Measured, R | Mixed-group share: set / all tasks (W1 clean run) |
|---|---|---|---|---|---|
| 1 | 55 | 0.147 | 0.1475 | 0.148 | 0.141 / 0.162 |
| 2 | 62 | 0.166 | 0.1655 | 0.156 | 0.151 / 0.169 |
| 3 | 57 | 0.152 | 0.1526 | 0.150 | 0.250 / 0.211 |

"Mixed" groups are those whose 16 rollouts do not all get the same honest reward. Only these carry a
GRPO gradient, so only these are affected by an inversion (an inverted uniform group still has zero
advantage). Most groups are uniform. Seed 2's set is about one point hotter than 15%, and seed 3's
set holds more learnable prompts than average.

Persistent task ids (MBPP `task_id`):

- seed 1: 602,604,612,635,637,648,657,660,665,670,675,678,679,684,696,699,700,708,713,714,720,724,729,730,734,740,754,766,767,775,781,796,801,830,834,863,874,879,883,892,896,898,899,918,921,923,934,935,937,948,959,960,962,970,974
- seed 2: 601,602,606,627,629,632,635,646,652,656,662,672,678,679,692,698,703,707,709,712,713,728,729,730,732,740,744,747,750,771,775,777,781,792,805,809,816,821,825,835,838,839,840,842,849,857,860,867,870,871,881,898,902,903,909,918,922,934,938,940,945,958
- seed 3: 606,607,611,622,623,633,637,638,646,650,662,674,693,694,716,721,722,734,737,745,747,770,771,779,787,799,807,832,841,843,844,848,856,863,867,873,874,890,892,897,901,902,903,906,924,930,931,934,935,937,940,944,949,952,960,962,968

**Disclosure:** R's results were known when this rule was written: V = 66.04 / 65.41 / 62.20,
mean 64.55, seed SD 2.06. No persistent run had started.

## Endpoint

The same as the W1 gate. For each run, **V** = the mean of the honest validation reward at verl steps
**240 and 260**, in percentage points (90 problems x 16 samples; -0.25 for no code block).

- `G = mean(V_R) - mean(V_P)`. **Positive G means persistent noise did more harm** (H2's direction).
- `s_p = sqrt((s_R^2 + s_P^2) / 2)`, where `s_R` and `s_P` are the sample SDs of V across the 3 seeds
  of each arm.
- `T = max(2 * s_p, 1.0)`.

## Outcomes, fixed in advance

1. **H2 SIGNAL: keep pushing H2.** `G >= T`.
2. **NO LARGE PERSISTENCE EFFECT: stop pushing H2. Add seeds, or pivot the headline to H3/H5.**
   `-T < G < T`. The proposal's End-M3 gate allows either response ("underpowered means more seeds or
   a pivot to H3/H5"); the Data analysis lane chooses.
3. **REVERSED: persistent noise did less harm than resampled, beyond seed noise.** `G <= -T`. H2 as
   stated is contradicted. Pivot as in 2, and bring it to the team as a result in its own right.

The 1.0-point floor in `T` is the Coding lane's addition to the sheet's rule. It only matters if
`s_p < 0.5`. Rationale: a gap smaller than the paper's own non-inferiority margin cannot carry a
headline. Strike it here if the Data analysis lane disagrees.

**Reported with the outcome but not used to decide it:** the one-sided 95% Welch lower bound on G;
the same rule on pass@1; per-seed differences (same task order per seed, but vLLM sampling differs
across pods, so these are not paired replicates); P against the W1 clean arm; the threshold
recomputed with the SD pooled over all three arms (clean, R, P; 6 degrees of freedom), and if the
outcome changes under it, the report says so; and the inverted share of *mixed* groups in each run.
The last is how much of the noise actually reached the gradient. Under persistent noise it can drift
away from 15% as the model learns, which is part of what persistence does and not something to
correct for.

## What the pilot can and cannot detect

Seed SD has been 1.6 (W1 clean) to 2.1 (W1 resampled) points, so `T` will probably be 3 to 4 points.
The table shows the simulated chance of each outcome: 3 seeds per arm, normal seed noise, equal SDs,
200,000 draws per cell. The true gap is the long-run R - P difference in points.

| True gap | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| H2 SIGNAL, seed SD 1.6 | 0.03 | 0.12 | 0.28 | 0.50 | 0.72 | 0.88 | 0.96 | 1.00 |
| H2 SIGNAL, seed SD 2.0 | 0.04 | 0.10 | 0.21 | 0.37 | 0.55 | 0.72 | 0.85 | 0.97 |
| REVERSED, seed SD 2.0 | 0.035 | 0.010 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

With no true difference, the rule says H2 SIGNAL about 4% of the time and REVERSED about 3.5%.
A true 2-point effect is missed about 4 times in 5. So outcome 2 means "no persistence effect big enough to lead the paper at 1.7B with 3 seeds". It does
**not** mean "persistence does not matter". That is the sheet's design (a cheap screen before the W4
grid), not a flaw introduced here.

Adding seeds does not lower this bar. `T` is twice the SD of a single run, not of the gap, so more seeds
only estimate `s_p` better. What more seeds shrink is the uncertainty of G itself. A Welch test with
80% power at one-sided 5% needs about `12.4 * (sigma / gap)^2` seeds per arm, plus one or two for the
t correction. At sigma = 1.84 (W1 pooled), that is about 3 to 4 seeds for a 4-point gap, 5 to 6 for
3 points, and 11 to 12 for 2 points. If the team takes the "more seeds" branch, that test, not `T`,
should decide.

## Validity checks (a failure voids the pilot rather than deciding it)

1. **Matched implementation.** All six manifests agree on `runner_script_sha256`, `team_repo_sha`,
   `reward_file_sha256`, `mbpp_scoring_sha256`, `val_parquet_sha256`, `verl_sha`,
   `verl_uv_lock_sha256`, the library versions (python, torch, vllm, ray, transformers, flash-attn,
   tensordict, numpy, datasets), the GPU model, every `cfg_*` setting except the seed and the two
   self-stop timers (`cfg_IDLE_STOP_MIN`, `cfg_MAX_RUNTIME_MIN`, which only decide when an idle pod
   stops), and every `MIXUP_*` knob except `MIXUP_MODE` and the log path. For each seed, both arms have
   the same `train_parquet_sha256`, and every run's mask seed equals its training seed. If the runner
   hash cannot be matched, the pilot is VOID by this rule unless the Data analysis lane accepts the
   difference in writing here. The report then quotes both hashes.
2. **Matched rate.** In every run, the inverted share of training groups is within 0.15 +/- 0.03
   (the W1 tolerance). Expected for P: 0.1475 / 0.1655 / 0.1526.
3. **Persistence happened, and only in P.** In every P run, each task has the same inverted status on
   every pass (a group made only of format failures is never inverted and is skipped), and the
   inverted set equals the set listed above for its mask seed. In every R run, at least 90% of tasks
   change status at least once (expected: about 99.5%).
4. **Honest validation.** No validation row was flipped, and every validation score equals the honest
   reward.
5. **Same reward scale.** Continuous in every run.
6. **Our numbers equal verl's own** (`analysis/log_to_wandb.py` verify).
7. **Complete, single-boot runs.** Exit 0, booted once, every step 1..260 logged, validation at 240
   and 260 present, exact per-sample record counts (`fleet.py` verify). There is no checkpoint-resume,
   so a run that dies or restarts is relaunched from step 0 under the same name, and only a complete
   run counts.
8. **Seeds 1, 2, 3 in both arms.**

**Not a check: whether the arms learn.** The pilot does not use the clean arm, so the open W1 "it
learns" question does not void it. If the Data analysis lane rules the W1 gate VOID because the
harness does not learn, hold this result as well.

## Report with the result

Per run: wall-clock hours, dollars (from the batch state), data centre, W&B link. These feed the W4
budget.

## State wherever the result is quoted

- R ran on 2026-10-13 and P runs in November, on different pods. Every W1 run already had its own pod,
  so pod-to-pod variation is inside the seed SD. Check 1 rules out software drift. Nothing rules out
  a host-level change at RunPod.
- The mask seed fixes which tasks are inverted. With 3 seeds the result depends on these three sets
  (see the mixed-group shares above).
- The W1 deviations from the paper all apply (`docs/w1_gate_threshold.md`). One rate (15%), one model
  family (Qwen3), one task (MBPP).

## How to apply

```
python analysis/w3_pilot.py --resampled RUNS/w1-resamp15-s1 RUNS/w1-resamp15-s2 RUNS/w1-resamp15-s3 \
                            --persistent RUNS/w3-persist15-s1 RUNS/w3-persist15-s2 RUNS/w3-persist15-s3 \
                            --clean RUNS/w1-clean-s1 RUNS/w1-clean-s2 RUNS/w1-clean-s3 \
                            --accept-gpu-diff      # persistent arm on A40, see "Deviation recorded before launch"
```

Exit 0 when every validity check passes (whatever the outcome), 1 when the pilot is VOID.
