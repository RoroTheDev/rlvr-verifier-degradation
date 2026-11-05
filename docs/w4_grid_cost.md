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
| Persistent noise | 4.97 | 4.47 | W3 pilot, 2 runs so far, at the cost if collected when done |

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
| A. As written | 108 | 501 | 555.67 | 639.02 |
| B. A, clean once | 81 | 399 | 442.83 | 509.25 |
| C. B with Qwen reuse | 72 | 357 | 396.52 | 456.00 |
| D. 2 families, 4 settings, 3 seeds, clean once, Qwen reuse | 45 | 224 | 248.91 | 286.25 |
| E. 3 families, 2 settings, 3 seeds, clean once, Qwen reuse | 36 | 175 | 193.91 | 223.00 |
| F. Qwen3-1.7B only, 4 settings, 3 seeds, Qwen reuse | 18 | 91 | 101.30 | 116.50 |
| G. Qwen3-1.7B only, 2 settings (baseline and one targeted), 3 seeds, Qwen reuse | 6 | 30 | 33.77 | 38.83 |

**Money needed:** the RunPod balance is about $7 (2026-11-05), so every option needs new funds.
Take the "with 15% allowance" column and add about $10 for the harness re-verification and smoke
runs. For example, option F needs about $127, option E about $233 and option A about $649.

**Wall-clock:** RunPod has granted several L40S pods at once. Six in parallel would run option F
in about 15 hours of GPU wall-clock, plus collection. Stock varies: on 2026-11-04 the two pilot
launches were refused as out of stock and went through on a retry a few minutes later.

## How these numbers are made

The Coding lane's local `grid_cost.py` reads the fleet state files (`w1-gate`, `w3-pilot-s1`,
`w3-pilot`). For a run that sat stopped before collection, it uses the cost the run would have had
if collected when done; the idle time is a controller problem, not a property of the cell
(`docs/w3_pilot_runs.md`). Rerun it with seed 2's cost once that run is collected; the persistent
row moves by about ±$0.50.
