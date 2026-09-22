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

More components will be added as the project progresses.
