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
directly, since verl's custom reward function
(`custom_reward_function.path`/`.name` in the config) is called per-response,
which fits a per-task persistence cache without changes. The TRL-specific
environment fixes from that prototype (version pins, patches) don't carry
over — they were specific to that integration path.

`llm-verifier-noise` stays vendored locally (gitignored) as the reference
implementation for the confusion-matrix noise formula being ported into
verl's reward function.

verl install and the ported reward function are in progress.

More components will be added as the project progresses.
