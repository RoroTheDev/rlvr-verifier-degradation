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

The pipeline treats MATH-500 `test` as the reference set and GSM8K `train` as
the candidate set to clean. It removes exact normalized question matches and
all candidates whose 5-token-shingle Jaccard similarity meets the configured
threshold. The similarity check is exact (not a probabilistic LSH lookup). It
writes the remaining GSM8K training rows to `clean_gsm8k_train.jsonl` and
counts removed and retained rows in `report.json`.

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
  `clean`, `resampled` or `persistent`; in `persistent` mode a task's flip is a
  hash of its identity, so every Ray worker and every epoch gets the same label
  with no shared state. Tests: `python reward_functions/test_persistence_reward.py`.
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
