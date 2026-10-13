# W1 gate: the six runs, their numbers, and how to reproduce them

**Status: numbers only.** The Coding lane produced and verified these runs. Calling the gate is the
Data analysis lane's decision, by the rule in `docs/w1_gate_threshold.md`. One validity check is open
(see "Open question"), so nothing here is tagged or called "replicated".

Runs finished 2026-10-13 (UTC). Qwen3-1.7B, thinking off, MBPP (374 train / 90 validation), 260 steps,
3 seeds per arm. W&B group `w1-gate`; how to read it: `docs/wandb_guide.md`.

## Endpoint per run

Validation reward (mean fraction of tests passed over 90 problems x 16 samples, -0.25 for no code block),
in percentage points. V is the mean of the two late evaluations.

| Run | Arm | Seed | Step 20 | Step 240 | Step 260 | V | V on pass@1 | Groups inverted | W&B |
|---|---|---|---|---|---|---|---|---|---|
| `w1-clean-s1` | clean | 1 | 56.38 | 62.16 | 65.49 | **63.83** | 56.91 | 0.000 | [ncfo2xjw](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/ncfo2xjw) |
| `w1-clean-s2` | clean | 2 | 59.68 | 63.54 | 62.29 | **62.92** | 56.11 | 0.000 | [ukdc3oil](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/ukdc3oil) |
| `w1-clean-s3` | clean | 3 | 56.77 | 60.27 | 61.18 | **60.72** | 53.47 | 0.000 | [8ptmw2l0](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/8ptmw2l0) |
| `w1-resamp15-s1` | noisy 15% | 1 | 53.34 | 67.05 | 65.03 | **66.04** | 59.51 | 0.148 | [zskysrpk](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/zskysrpk) |
| `w1-resamp15-s2` | noisy 15% | 2 | 54.06 | 65.13 | 65.68 | **65.41** | 59.20 | 0.156 | [3bg4ogga](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/3bg4ogga) |
| `w1-resamp15-s3` | noisy 15% | 3 | 54.10 | 61.42 | 62.99 | **62.20** | 56.39 | 0.150 | [sny6pyv8](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/sny6pyv8) |

| | Mean V | Seed SD (n=3) |
|---|---|---|
| Clean | 62.49 | 1.59 |
| Noisy 15% | 64.55 | 2.06 |

- Gap D = clean - noisy = **-2.06 points** (the noisy arm is higher).
- Pooled seed SD 1.84; standard error of the gap 1.50; Welch df 3.77.
- One-sided 95% upper bound on D: **+1.20 points**.
- On pass@1 instead of reward: D = -2.87, upper bound +0.20, pooled SD 1.76.

Against the rule's thresholds: D <= 1.0 holds and the upper bound <= 2.0 holds.

## Open question (decides between "void" and the rule's outcome)

The threshold doc's validity check "it learns" reads: in every clean seed, "V at step 260 exceeds the
validation reward at step 20 by at least 3 points". V is defined as the mean of steps 240 and 260, so the
sentence can mean either the step-260 value or the endpoint V. The two readings can disagree:

| Clean run | Step 20 | Step 260 | Gain (step 260 alone) | V | Gain (V) |
|---|---|---|---|---|---|
| seed 1 | 56.38 | 65.49 | +9.11 | 63.83 | +7.45 |
| seed 2 | 59.68 | 62.29 | **+2.62** | 62.92 | +3.24 |
| seed 3 | 56.77 | 61.18 | +4.42 | 60.72 | +3.96 |

`analysis/w1_gate.py` implements the step-260 reading. Its output for these runs: `OUTCOME: VOID` (it notes that the rule would otherwise say `REPLICATED`).
This was flagged before the noisy runs finished but not resolved before they did, so whoever owns the rule
is now choosing the reading with the gap known. That should be stated wherever the result is quoted.

Every other validity check passes for all six runs: noise fraction within 0.15 +/- 0.03 in each noisy run,
zero inverted groups in each clean run, no validation row flipped, both arms on the continuous reward
scale, our numbers equal to verl's own on all 260 steps and all 26 validation values per run, same seeds
in both arms, one code commit.

## Things the numbers do not say

- Three seeds per arm. The gap is 1.4 standard errors from zero.
- Within one run the two late evaluations differ by up to 3.3 points, so single evaluations are noisy.
- Runs are not bit-for-bit repeatable: two pods with the same seed drew slightly different first-step
  samples. A rerun of a seed will be close, not identical.
- Seeds share the data order across arms, which the unpaired test ignores.

## Training behaviour, for the analysis lane (not interpreted here)

Mean response length in tokens and policy entropy, first 10 steps -> last 10 steps:

| Run | Response length | Entropy |
|---|---|---|
| `w1-clean-s1` | 184 -> 79 | 0.104 -> 0.028 |
| `w1-clean-s2` | 198 -> 72 | 0.111 -> 0.028 |
| `w1-clean-s3` | 219 -> 117 | 0.117 -> 0.038 |
| `w1-resamp15-s1` | 228 -> 346 | 0.120 -> 0.179 |
| `w1-resamp15-s2` | 237 -> 252 | 0.119 -> 0.144 |
| `w1-resamp15-s3` | 219 -> 100 | 0.114 -> 0.052 |

## Cost and wall-clock per run (for the W4 budget)

Each on one NVIDIA L40S (48 GB) at $1.09/hr plus about $0.02/hr disk, RunPod secure cloud.

| Run | Data centre | Wall-clock | Median s/step | Cost |
|---|---|---|---|---|
| `w1-clean-s1` | EU-NL-1 | 3.72 h | 43 | $4.13 |
| `w1-clean-s2` | EUR-IS-2 | 3.53 h | 39 | $3.91 |
| `w1-clean-s3` | SE | 4.05 h | 50 | $4.50 |
| `w1-resamp15-s1` | SE | 6.72 h | 80 | $7.46 |
| `w1-resamp15-s2` | EUR-IS-2 | 5.84 h | 72 | $6.48 |
| `w1-resamp15-s3` | SE | 4.43 h | 59 | $4.92 |

Sum of the per-run estimates $31.41; billed for the batch (balance before minus after) **$31.32**.
Clean runs averaged $4.18 and 3.8 h; noisy runs $6.29 and 5.7 h.
The same setting measured about $11.9 per run on an A100, and about $8.1 on an L40S without dynamic
micro-batching (both from 4-step probes).

## Reproduce

Code: this repo at `ca336b32c606e56eae9625984f6c1ef1e9c56c72`; verl at
`8718ca30a3f002f93b7c4fd99b9b2506718681bc`. Each run's manifest (in its W&B config under `manifest/*`)
records these, the file hashes and every setting.

```
python runpod/launch.py launch --run-name w1-clean-s1 --gpu "NVIDIA L40S" \
  --ref ca336b32c606e56eae9625984f6c1ef1e9c56c72 \
  --env VERL_REF=8718ca30a3f002f93b7c4fd99b9b2506718681bc \
  --env TASK=mbpp --env MODEL=Qwen/Qwen3-1.7B --env ENABLE_THINKING=0 \
  --env MIXUP_MODE=clean --env MIXUP_REWARD_SCALE=continuous --env SEED=1 \
  --env DYN_BSZ=True --env MAX_TOKENS_PER_GPU=8192 \
  --env WANDB_RUN_GROUP=w1-gate --env WANDB_TAGS=w1-gate,qwen3-1.7b,nothink,mbpp \
  --env IDLE_STOP_MIN=45 --env MAX_RUNTIME_MIN=720
```

- Seeds 1, 2, 3. Noisy arm: `--env MIXUP_MODE=group_resampled --env MIXUP_GROUP_P=0.15` (run names
  `w1-resamp15-s<seed>`); everything else identical.
- `MIXUP_REWARD_SCALE=continuous` is required on the clean arm. Clean mode defaults to a pass/fail reward
  while the group-noise modes use the fraction of tests passed; without it the arms are on different scales.
- After a run: `launch.py fetch`, then `python analysis/log_to_wandb.py <run dir> --push`, then
  `launch.py stop`. Then:

```
python analysis/w1_gate.py --clean w1-clean-s1 w1-clean-s2 w1-clean-s3 \
                           --noisy w1-resamp15-s1 w1-resamp15-s2 w1-resamp15-s3
```

The fetched run directories (per-sample logs, about 90 MB each) are not in the repo.

## Deviations from Plesner et al. (arXiv 2604.07666 v2)

Qwen3-1.7B (paper: 8B primary, 4B-9B across families); thinking off; verl instead of their framework;
8 optimizer updates per step inferred from "global batch size 96"; MBPP prompt wording approximated;
evaluation responses capped at 4096 tokens (paper: 8192); 3 seeds per arm; one noise rate (15%);
one L40S GPU per run; dynamic micro-batching (8192 tokens), which changes how a mini-batch is split
for memory, not the gradient (verl normalises the loss by the mini-batch size).

Data: the 374 / 90 split is MBPP `full` (`google-research-datasets/mbpp`) taken straight from Hugging Face by
`data_prep/mbpp_prep.py`. It was **not** screened for contamination: the Data lane's decontamination pipeline
covers GSM8K against MATH-500, not MBPP. The project rule is "decontaminate, and say so"; for this gate the
honest statement is that no MBPP screen was run.

## Script output, verbatim

```
W1 gate: mean validation reward at verl steps 240 and 260, percentage points

| run | arm | seed | step 240 | step 260 | V | V (pass@1) | inverted groups | W&B |
|---|---|---|---|---|---|---|---|---|
| w1-clean-s1 | clean | 1 | 62.16 | 65.49 | 63.83 | 56.91 | 0.000 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/ncfo2xjw |
| w1-clean-s2 | clean | 2 | 63.54 | 62.29 | 62.92 | 56.11 | 0.000 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/ukdc3oil |
| w1-clean-s3 | clean | 3 | 60.27 | 61.18 | 60.72 | 53.47 | 0.000 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/8ptmw2l0 |
| w1-resamp15-s1 | noisy | 1 | 67.05 | 65.03 | 66.04 | 59.51 | 0.148 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/zskysrpk |
| w1-resamp15-s2 | noisy | 2 | 65.13 | 65.68 | 65.41 | 59.20 | 0.156 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/3bg4ogga |
| w1-resamp15-s3 | noisy | 3 | 61.42 | 62.99 | 62.20 | 56.39 | 0.150 | https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/sny6pyv8 |

clean  mean 62.49, seed SD 1.59 (n=3)
noisy  mean 64.55, seed SD 2.06 (n=3)
gap D = clean - noisy = -2.06 pt; pooled seed SD s_p = 1.84; SE = 1.50; Welch df = 3.77
one-sided 95% upper bound on D: +1.20 pt  (paper's margin 1.0, bound limit 2.0, 2 x s_p = 3.68)
secondary, on pass@1: D = -2.87 pt, UB = +0.20 pt, s_p = 1.76

validity checks:
  [ok] w1-clean-s1: learns (56.38 at step 20 -> 65.49 at step 260, gain +9.11 pt, need >= 3.0)
  [ok] w1-clean-s1: clean arm has no inverted groups (0.0000)
  [FAIL] w1-clean-s2: learns (59.68 at step 20 -> 62.29 at step 260, gain +2.62 pt, need >= 3.0)
  [ok] w1-clean-s2: clean arm has no inverted groups (0.0000)
  [ok] w1-clean-s3: learns (56.77 at step 20 -> 61.18 at step 260, gain +4.42 pt, need >= 3.0)
  [ok] w1-clean-s3: clean arm has no inverted groups (0.0000)
  [ok] w1-resamp15-s1: inverted 0.1481 of prompt groups (configured 0.15, tolerance 0.03)
  [ok] w1-resamp15-s2: inverted 0.1563 of prompt groups (configured 0.15, tolerance 0.03)
  [ok] w1-resamp15-s3: inverted 0.1502 of prompt groups (configured 0.15, tolerance 0.03)
  [ok] w1-clean-s1: validation was scored honestly (0 flipped rows)
  [ok] w1-clean-s1: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-clean-s1: our numbers equal verl's own
  [ok] w1-clean-s2: validation was scored honestly (0 flipped rows)
  [ok] w1-clean-s2: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-clean-s2: our numbers equal verl's own
  [ok] w1-clean-s3: validation was scored honestly (0 flipped rows)
  [ok] w1-clean-s3: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-clean-s3: our numbers equal verl's own
  [ok] w1-resamp15-s1: validation was scored honestly (0 flipped rows)
  [ok] w1-resamp15-s1: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-resamp15-s1: our numbers equal verl's own
  [ok] w1-resamp15-s2: validation was scored honestly (0 flipped rows)
  [ok] w1-resamp15-s2: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-resamp15-s2: our numbers equal verl's own
  [ok] w1-resamp15-s3: validation was scored honestly (0 flipped rows)
  [ok] w1-resamp15-s3: reward scale ['continuous'] (both arms must be continuous)
  [ok] w1-resamp15-s3: our numbers equal verl's own
  [ok] at least 3 seeds per arm (3 clean, 3 noisy)
  [ok] same distinct seeds in both arms (clean ['1', '2', '3'], noisy ['1', '2', '3'])
  [ok] one code commit for every run (['ca336b32c6'])

OUTCOME: VOID (a validity check failed; the rule would have said: REPLICATED)
```
