# rlvr-verifier-degradation

This repository contains the setup for studying verifier degradation in
RLVR/GRPO training. It currently includes a dataset decontamination pipeline
that compares the GSM8K training split with the MATH-500 test split.

## Setup

Install the base dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional Qwen semantic judge requires:

```bash
python -m pip install -r requirements-qwen.txt
```

## Dataset decontamination

Run the GSM8K/MATH-500 pipeline:

```bash
python -m decontamination_pipeline.decontamination_pipeline \
  --output-dir results/gsm8k_math500 \
  --threshold 0.80
```

The pipeline performs exact matching and MinHash/LSH near-duplicate matching.
It writes `clean_gsm8k_train.jsonl` and `report.json` to the selected output
directory.

Qwen is optional:

```bash
python -m decontamination_pipeline.decontamination_pipeline \
  --use-qwen --qwen-limit 100
```

Qwen judges supplied text pairs only; it does not reveal Qwen's private
pretraining data.

## GRPO training (Coding lane)

GRPO training targets [verl](https://github.com/volcengine/verl), not `trl` —
verl's throughput and multi-GPU support are needed for the full compute grid
and the project's two planned extension tasks. An earlier prototype built
directly on `eth-sri/llm-verifier-noise` (TRL-based) proved out the
persistence-flag design and the GPU/RunPod setup; that design carries over
directly, since verl's custom reward function is called per-response. The
TRL-specific environment fixes from that prototype (version pins, patches) don't
carry over — they were specific to that integration path.

`llm-verifier-noise` stays vendored locally (gitignored) as the reference
implementation for the confusion-matrix noise formula that was ported.

- `reward_functions/persistence_reward.py` — the noisy verifier. `MIXUP_MODE` is
  `clean`, `resampled`, `persistent`, `group_resampled` or `group_persistent`.
  Flips are hashes of a task's identity, so every Ray worker gets the same answer
  with no shared state. The `group_*` modes are Plesner et al.'s headline noise
  (with probability `MIXUP_GROUP_P` a prompt's whole group of outcomes is
  inverted): `group_resampled` redraws the coin each time a prompt is seen,
  `group_persistent` fixes it for the whole run — same rate, same structure, only
  persistence differs. Noise applies to training rows only; validation is always
  scored honestly. Tests: `python reward_functions/test_persistence_reward.py`.
- `reward_functions/mbpp_scoring.py` + `data_prep/mbpp_prep.py` — MBPP scorer
  (fraction of unit tests passed, -0.25 for no code block) and the converter that
  writes verl-format MBPP parquet, with an `encounter` number per row so the noise
  can be redrawn per pass. `--env TASK=mbpp` is a preset matching Plesner et al.
  (arXiv 2604.07666 v2, Table 3): 48 prompts x 16 rollouts, lr 1e-6, clip 0.2/0.28,
  no KL, per-response token-mean loss, Adam wd 0.1 / beta2 0.98, 8 optimizer updates
  per step, 260 steps with validation every 20 (the paper's late evals are steps
  240/260 in verl numbering). Anything passed with `--env` overrides the preset, and
  `manifest.txt` records the effective value of every knob. `--buckets file.csv`
  (task_id plus any columns) adds labels to `extra_info` so noise can be targeted
  with `MIXUP_TARGET_FIELD/OP/VALUE`.
- `analysis/run_metrics.py` — per-step and per-validation-round metrics from a run's
  `mixup_logs/` (honest pass@1 / pass@k, within-group reward variance, the GRPO
  advantage denominator, zero-variance group fractions, inversion rate, per-bucket
  rows). On a real run its `noisy_reward` equals verl's own `critic/score/mean`.
  Tests: `python analysis/test_run_metrics.py`.
- `runpod/launch.py` + `runpod/verl_run.sh` — one-command GRPO run on a RunPod
  pod (`launch`, `watch`, `fetch`, `stop`); every run writes a `manifest.txt`
  with the repo, verl and package versions that produced it.

**Wiring gotcha:** the reward must be set as `reward.custom_reward_function.path`
/ `.name`. The bare `custom_reward_function.*` key that verl's docs suggest is
silently ignored by the active reward manager (verl falls back to its default
scoring with no error). Check the resolved config at the top of the training log.

**Status:** a 2-step Qwen2.5-0.5B / GSM8K smoke run completes with the persistent
reward active, and a task's label is identical across rollouts scored by
different Ray workers. verl is not pinned to a release yet — pass
`--env VERL_REF=<sha>` for any run whose numbers will be reported.

More components will be added as the project progresses.
