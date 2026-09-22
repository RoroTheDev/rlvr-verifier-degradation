# rlvr-verifier-degradation

Overview
This repository contains the early setup and structure for our RLVR verifier degradation project.
The goal is to study how verifier error — including rate, asymmetry, and persistence — affects RLVR/GRPO training stability across different regimes.

This repo will hold:

the noise model implementation

persistence masks

GRPO training scripts

experiment configs

logs and analysis outputs

More components will be added as the project progresses.

## Dataset decontamination

`decontamination_pipeline/decontamination_pipeline.py` compares the GSM8K
training split against the MATH-500 test split:

```bash
python -m decontamination_pipeline.decontamination_pipeline \
  --output-dir results/gsm8k_math500 \
  --threshold 0.80
```

Every run writes `results/clean_gsm8k_train.jsonl` and
`results/report.json`. Exact matching and MinHash/LSH are enabled by default.
Qwen is optional:

```bash
python -m decontamination_pipeline.decontamination_pipeline \
  --use-qwen --qwen-limit 100
```

Install `requirements-qwen.txt` only for Qwen mode. Qwen judges text pairs; it
does not reveal Qwen's private pretraining data.

Work Log
2026‑08‑30
Created the GitHub repository

Added initial project structure
