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
  `manifest.txt` records the effective value of every knob. Targeted noise:
  `--env BUCKETS=<csv in this repo>` labels tasks and `MIXUP_TARGET_FIELD/OP/VALUE`
  selects which labels get noise; the file format is in `docs/bucket_file.md`.
- `analysis/run_metrics.py` — per-step and per-validation-round metrics from a run's
  `mixup_logs/` (honest pass@1 / pass@k, within-group reward variance, the GRPO
  advantage denominator, zero-variance group fractions, inversion rate, the verifier's
  false-positive / false-negative rates as the trainer saw them, per-bucket rows). On a
  real run its `noisy_reward` equals verl's own `critic/score/mean`.
  Tests: `python analysis/test_run_metrics.py`.
- `analysis/log_to_wandb.py` — checks those numbers against verl's own, then adds them
  and the run manifest to the run's W&B run (`docs/wandb_guide.md` explains the keys).
  `analysis/w1_gate.py` applies the rule in `docs/w1_gate_threshold.md`; the six gate
  runs, their numbers and how to reproduce them are in `docs/w1_gate_runs.md`.
- `runpod/launch.py` + `runpod/verl_run.sh` — one-command GRPO run on a RunPod
  pod (`launch`, `watch`, `fetch`, `stop`, `start`, `pods`); every run writes a
  `manifest.txt` with the repo, verl and package versions that produced it. A pod bills
  until it is terminated: pass `--env IDLE_STOP_MIN=45 --env MAX_RUNTIME_MIN=720` so a
  pod nobody collects stops itself.

**Wiring gotcha:** the reward must be set as `reward.custom_reward_function.path`
/ `.name`. The bare `custom_reward_function.*` key that verl's docs suggest is
silently ignored by the active reward manager (verl falls back to its default
scoring with no error). Check the resolved config at the top of the training log.

**Status (2026-10-13):** the six Week 1 gate runs (Qwen3-1.7B, thinking off, MBPP;
3 clean seeds and 3 with 15% resampled group noise) are complete and verified against
verl's own numbers. Their endpoints, cost, the open validity question and the exact
reproduction command are in `docs/w1_gate_runs.md`; calling the gate is the Data
analysis lane's decision. Persistent and targeted noise are implemented and tested
(a task's label is identical across epochs and across Ray workers). Pass `--ref <sha>`
and `--env VERL_REF=<sha>` for any run whose numbers will be reported.

More components will be added as the project progresses.
