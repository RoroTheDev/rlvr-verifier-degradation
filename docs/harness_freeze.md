# Harness freeze, version 2 (W3)

**Status: tagged `w3-freeze` = `685dfe9` (2026-11-12).** The re-verification run below passed on
that commit: `w3-freeze-check`, 2026-11-12, L40S, W&B `1hbry75f`, $0.55. The tag message lists the
pins and the results. A first attempt on 2026-11-05 finished, but its pod was stuck stopped for a
week and was terminated uncollected. Commits after `685dfe9` change docs only.
Version 1 is tag `w1-freeze`. Every manifest from this version on records `harness_version=2`.

## Pins

| What | Pin |
|---|---|
| Runner `runpod/verl_run.sh`, sha256 as the pod records it (`runner_script_sha256`, LF line endings) | `cb0d91fd27331c9a34d1ce20efd4219a05403c00b6d9ec8845b1f40496014b05` |
| Reward `reward_functions/persistence_reward.py` | `869b54ddbc9d06e79587b526b66b4ff2aa5a133901d0fab7291037c79f7017e5` |
| Scorer `reward_functions/mbpp_scoring.py` | `c63e5eb98b63663959f78a10baeeeadd5cc63bc7c46c8ed8c6022be3aadbca95` |
| Data builder `data_prep/mbpp_prep.py` | `bf2c41456df3a6f78f05bf99106fed8dc6999aabafbc40e4347bd559e4be39c8` |
| verl | `8718ca30a3f002f93b7c4fd99b9b2506718681bc` (pass `--env VERL_REF=<this>`) |
| verl `uv.lock` | `7f4be8c9a89a2ecc53d720583e376749aa618cba6a86faca66dcea6566ed94f7` |
| Stack | python 3.12.3, torch 2.13.0+cu130, vllm 0.29.0, ray 2.55.1, transformers 5.12.1, flash-attn 2.8.3 |
| Image | `runpod/pytorch:1.2.0-rc.162-cu1290-torch291-ubuntu2404` |
| GPU | L40S, NVIDIA driver ≥ 580 (seen: 580.126.09, 595.91.07) |

The stack row is what `uv.lock` resolves to, as recorded in the W3 pilot manifests.

## What changed since the gate runs (tag `w1-gate-runs`, 08e892e)

- **Targeted noise.** `BUCKETS=<csv in this repo>` writes a label per task into the data. Then
  `MIXUP_TARGET_FIELD` / `_OP` / `_VALUE` choose which tasks get noise. A task with no label is never
  targeted, and text labels are compared as text. A missing bucket file stops the run with exit 4;
  a failed data build stops it with exit 5.
- **Harness version in every manifest** (`harness_version=2`).
- **Launcher host filter off by default** (`runpod/launch.py --cuda-versions ""`). RunPod's
  `allowedCudaVersions=["13.0"]` refused every GPU type on 2026-11-03. The runner's preflight still
  stops a pod on an old driver with `CUDA_UNAVAILABLE`.

Untargeted runs compute the same thing as the gate: same reward, scorer, data order and settings.
The runner hash differs, so cross-version comparisons must use the `--accept-runner-diff` path the
pilot script already has, and say so.

## Re-verification before tagging

The W2 targeted smoke recipe, run on the commit to be tagged: run `w3-freeze-check` on L40S with
Qwen3-1.7B and MBPP.

- **Length:** `STEPS=16`, `MBPP_PASSES=3`, `VAL_FREQ=16`. The MBPP preset otherwise sets 260 steps.
- **Noise:** `group_persistent` at p = 0.5, targeted at `bucket=odd` with the smoke bucket file
  `data_prep/buckets/smoke_parity.csv`. p = 0.5 inverts enough targeted groups for a sharp check.
- **Cost:** the W2 run cost $0.48.

It passes when:

- the run exits 0, booted once, with every step present and exact per-sample record counts (the
  `fleet.py` fetch check);
- the manifest says `harness_version=2`, `team_repo_sha` is the commit to be tagged, and the runner,
  reward and scorer hashes and every library version equal the pins above;
- the targeted check passes: only targeted tasks are inverted, each with one status on every pass,
  and validation is untouched;
- `analysis/log_to_wandb.py` finds our numbers equal to verl's.

Then tag it, with the pins in the tag message as `w1-freeze` has them:

```
git tag -a w3-freeze <commit> -m "<pins and verification run>"
git push origin w3-freeze
```

## Known limits of this version

- **No checkpoint-resume:** a run that dies restarts from step 0 (W4 item 2 in `docs/w4_plan.md`).
- **Collection needs the pod or an awake controller.** A finished pod stops itself after
  `IDLE_STOP_MIN` and can restart only on its own host. In W3 that took 11 to 20 hours (W4 item 1).
- **Step time varies 44 to 80 s/step across L40S hosts.** Wall-clock is not comparable between
  runs.
- **vLLM sampling is not bit-reproducible.** Two runs with the same seed agree in distribution, not
  sample by sample.
- **The W3 persistent arm did not run on this version.** It ran on the gate's runner, so it matches
  W1.
