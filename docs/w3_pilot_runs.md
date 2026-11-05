# W3 pilot: the persistent runs, their numbers and their cost

**Status: numbers only; seed 2 not yet collected.** The Coding lane produced and verified these runs.
The Data analysis lane calls the pilot, using the rule in `docs/w3_pilot_threshold.md`, which was
fixed before any persistent run. This file computes no gap. Seed 2 trained to step 260, as its W&B
run shows, but its files were still on its stopped pod when this was written ("Collection" below).

Setup as fixed in the rule:

- **R** = the three W1 resampled runs.
- **P** = `w3-persist15-s1..s3`.
- P differs from R only in the noise mode (`group_persistent`, p = 0.15). Same seeds, the gate's
  runner (worktree of tag `w1-gate-runs`, runner sha256 `3951eddc...`), team repo at `ca336b3`, and
  L40S on RunPod secure cloud.
- Qwen3-1.7B with thinking off, MBPP (374 train / 90 validation), 260 steps. W&B group `w3-pilot`.

## Endpoint per run

Validation reward in percentage points: mean fraction of tests passed over 90 problems × 16
samples, with −0.25 when there is no code block. V is the mean of steps 240 and 260.

| Run | Arm | Seed | Step 240 | Step 260 | V | V on pass@1 | Groups inverted | Tasks inverted | W&B |
|---|---|---|---|---|---|---|---|---|---|
| `w1-resamp15-s1` | resampled | 1 | 67.05 | 65.03 | **66.04** | 59.51 | 0.1481 | 371 of 374 | [zskysrpk](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/zskysrpk) |
| `w1-resamp15-s2` | resampled | 2 | 65.13 | 65.68 | **65.41** | 59.20 | 0.1563 | 374 of 374 | [3bg4ogga](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/3bg4ogga) |
| `w1-resamp15-s3` | resampled | 3 | 61.42 | 62.99 | **62.20** | 56.39 | 0.1502 | 373 of 374 | [sny6pyv8](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/sny6pyv8) |
| `w3-persist15-s1` | persistent | 1 | 59.07 | 58.40 | **58.74** | 52.74 | 0.1475 | 55 of 374 | [guh6zkve](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/guh6zkve) |
| `w3-persist15-s2` | persistent | 2 | pending | pending | pending | pending | pending | pending | [zeinyn60](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/zeinyn60) |
| `w3-persist15-s3` | persistent | 3 | 59.83 | 59.81 | **59.82** | 54.86 | 0.1526 | 57 of 374 | [rjvpurqw](https://wandb.ai/yourlocalfifa-university-at-albany/rlvr-verifier-degradation/runs/rjvpurqw) |

"Tasks inverted" counts tasks inverted at least once. In R almost every task is hit at some point
over the 34 passes. In P each seed's set is fixed in advance, and it matches the rule doc's
precomputed sets: 55 tasks for seed 1, 62 for seed 2, 57 for seed 3.

## Checks on the collected persistent runs

Seeds 1 and 3 each pass every check below against the same-seed gate run (the Coding lane's local
`persist_check.py` on the full run, then `analysis/log_to_wandb.py`):

- **Matched implementation.** Runner, team repo, reward, scorer, verl, uv.lock and validation
  parquet hashes are equal. So are python, torch, vllm, ray, transformers and flash-attn, the
  per-seed train parquet and mask seed, and every setting except the two self-stop timers. The GPU
  model is the same; only the host driver differs (595.91 vs 580.126).
- **Persistence.** Every one of the 374 tasks has one status on every pass. The inverted tasks are
  exactly the predicted set, with none outside it. The inverted share is 0.1475 for seed 1 and
  0.1526 for seed 3, matching the predictions exactly.
- **Honest validation.** Zero validation samples were flipped, and our honest reward and pass@1
  equal verl's own numbers at all 13 evaluations (difference ≤ 3e-9).
- **Complete, single-boot runs.** Exit 0, one boot, steps 1–260 all present, and exactly 199,680
  train and 18,720 validation per-sample records.

The full validity check (all eight checks, both arms) is `analysis/w3_pilot.py`. It runs once seed
2 is in; the command is below.

## Wall-clock and dollars per cell

| Run | GPU, data centre | s/step (median) | Launch to done | $ billed | $ if collected at done |
|---|---|---|---|---|---|
| `w1-resamp15-s1` | L40S, SE | 79.5 | 6.72 h | 7.46 | 7.46 |
| `w1-resamp15-s2` | L40S, EUR-IS-2 | 71.7 | 5.84 h | 6.48 | 6.48 |
| `w1-resamp15-s3` | L40S, SE | 59.2 | 4.43 h | 4.92 | 4.92 |
| `w3-persist15-s1` | L40S, SE | 43.9 | 3.67 h | 5.40 | 4.07 |
| `w3-persist15-s2` | L40S, OC-AU-1 | about 73 | about 6 h | pending | about 6.6 |
| `w3-persist15-s3` | L40S, US-TX-4 | 66.0 | 5.28 h | 8.23 | 5.86 |

- **Billed vs at done.** "$ billed" adds the idle self-stop window, the restart for collection and
  storage while stopped. "$ if collected at done" is what the run costs when a controller collects it
  as soon as it finishes; use that column for planning (`docs/w4_grid_cost.md`).
- **Host speed varies.** The same GPU model ran at 44 to 80 s/step on different hosts, so
  wall-clock for an identical run varies by up to 1.8×. The step-time difference between arms is not
  a property of the noise mode.
- **Pilot spend.** $22.41 billed by 2026-11-05 03:58 PST, against $22.00 authorised, plus roughly
  $0.50 to collect the two runs that sat stopped. The overrun is idle time while the controlling PC
  slept: $1.33 for seed 1, $2.37 for seed 3 and about $2.40 for seed 2, so about $6 in all.

## What went wrong, and what changed

1. **Every launch was refused at first.** The launcher's default host filter,
   `allowedCudaVersions=["13.0"]`, is the highest value the API accepts and matched no available
   host. Launching without it got an L40S at once. `runpod/launch.py` now defaults to no filter; the
   runner's preflight still stops a pod on an old driver with `CUDA_UNAVAILABLE`.
2. **The PC slept through the end of every run.** Each pod stopped itself as designed, but a stopped
   pod restarts only on its own host. The hosts had no free GPU for 11 to 20 hours, so collection
   waited and idle time was billed. No data was lost. The fix is W4 work (`docs/w4_plan.md`): upload
   a run's files off the pod at STAGE:DONE, so collecting a run never depends on restarting its pod.
   Until then, the controlling PC must stay awake until each run is collected.
3. **`IDLE_STOP_MIN=120` for seeds 2 and 3.** The longer window was meant to keep pods reachable
   through a sleep, but the sleep lasted 5 hours. It only added idle billing. Back to 45 from now on.
4. **`analysis/w3_pilot.py` crashed when an arm had fewer than 2 seeds.** It now reports VOID with
   the reason. A test and a mutation cover this.

## Collection and how to finish

Seed 2's pod (`ouf31ck9mfu3dr`, OC-AU-1) is stopped with the run's files on its volume. A local
collector retries the restart every 10 minutes. When the pod comes back, the collector:

1. fetches, verifies and terminates it;
2. runs the same checks as above and pushes the honest metrics to W&B;
3. runs the pilot script:

```
python analysis/w3_pilot.py --resampled RUNS/w1-resamp15-s1 RUNS/w1-resamp15-s2 RUNS/w1-resamp15-s3 \
                            --persistent RUNS/w3-persist15-s1 RUNS/w3-persist15-s2 RUNS/w3-persist15-s3 \
                            --clean RUNS/w1-clean-s1 RUNS/w1-clean-s2 RUNS/w1-clean-s3 \
                            --state w1-gate.state.json w3-pilot-s1.state.json w3-pilot.state.json
```

The output goes to the Data analysis lane, which calls the pilot. Fill in seed 2's rows above from
that output.
