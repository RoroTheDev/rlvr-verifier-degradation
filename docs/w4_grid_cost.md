# W4 grid: what it costs, and pruned options

**Status: planning numbers from the Coding lane, 2026-11-05.** Which option runs is a team decision.
It also depends on the W3 pilot call (`docs/w3_pilot_threshold.md`): if the pilot says "no large
persistence effect", the headline moves to H3/H5 and the grid should be cut to match.

## Cost per cell (measured)

All measured on L40S (RunPod secure cloud), Qwen3-1.7B, MBPP, 260 steps, at $1.09–1.11/hr.

| Regime | $ per run | Hours per run | From |
|---|---|---|---|
| Clean | 4.18 | 3.77 | W1 gate, 3 runs |
| Resampled noise | 6.29 | 5.67 | W1 gate, 3 runs |
| Persistent noise | 5.56 | 5.01 | W3 pilot, 3 runs, at the cost if collected when done |

- Per-run cost varies by host: the same GPU model ran at 44 to 80 s/step. Three runs per regime
  pin the mean only to about ±20%.
- **The other families (GLM4, Llama 3.1) are unmeasured.** These numbers assume they cost the same
  per run as Qwen3-1.7B, which holds only at a similar size and the same task.
- The 15% allowance pays for relaunches. With no checkpoint-resume, a run that dies is paid twice
  (`docs/w4_plan.md`).

## Options

The proposal's grid is 3 regimes × 4 coverage/targeting settings × 3 families × 3 seeds. "Clean
once" counts one clean run per family and seed, since coverage and targeting do not apply to a
clean verifier. "Qwen reuse" counts the 9 finished Qwen3-1.7B baseline cells (W1 clean and
resampled, W3 persistent) as done. Reuse needs the team's agreement, because the frozen W4 harness
(`docs/harness_freeze.md`) will not be the gate's runner byte for byte.

| Option | New cells | GPU-hours | $ | $ with 15% allowance |
|---|---|---|---|---|
| A. As written | 108 | 520 | 577.00 | 663.55 |
| B. A, clean once | 81 | 418 | 464.15 | 533.78 |
| C. B with Qwen reuse | 72 | 375 | 416.07 | 478.48 |
| D. 2 families, 4 settings, 3 seeds, clean once, Qwen reuse | 45 | 235 | 261.35 | 300.56 |
| E. 3 families, 2 settings, 3 seeds, clean once, Qwen reuse | 36 | 183 | 202.80 | 233.22 |
| F. Qwen3-1.7B only, 4 settings, 3 seeds, Qwen reuse | 18 | 96 | 106.63 | 122.63 |
| G. Qwen3-1.7B only, 2 settings (baseline and one targeted), 3 seeds, Qwen reuse | 6 | 32 | 35.54 | 40.88 |

**Money needed:** the RunPod balance is about $6 (2026-11-05), so every option needs new funds.
Take the "with 15% allowance" column and add about $10 for the harness re-verification and smoke
runs. For example, option F needs about $133, option E about $243 and option A about $674.

**Wall-clock:** RunPod has granted several L40S pods at once. Six in parallel would run option F
in about 16 hours of GPU wall-clock, plus collection. Stock varies: on 2026-11-04 the two pilot
launches were refused as out of stock and went through on a retry a few minutes later.

## How these numbers are made

The Coding lane's local `grid_cost.py` reads the fleet state files (`w1-gate`, `w3-pilot-s1`,
`w3-pilot`). For a run that sat stopped before collection, it uses the cost the run would have had
if collected when done; the idle time is a controller problem, not a property of the cell
(`docs/w3_pilot_runs.md`). All three persistent seeds are in.
